import numpy as np
import torch
from torch import nn
from typing import Dict, List
from reRLs.policies.base_policy import GaussianPolicy
from reRLs.infrastructure.utils import pytorch_util as ptu
from contextlib import contextmanager

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y

def compute_weight_decay(weight_decay, model_params: np.ndarray):
    model_params = np.array(model_params)
    return - weight_decay * np.mean(model_params * model_params, axis=1)

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def _compute_step(self, globalg):
        raise NotImplementedError

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step

class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step

class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

class GaussianPolicyES(GaussianPolicy):

    def __init__(
        self,
        obs_dim,
        act_dim,
        layers: List=[64,64],
        learning_rate: int=1e-3,
        discrete=True,
        es_config: Dict=None
    ):

        nn.Module.__init__(self)
        # init params
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layers = layers
        self.discrete = discrete
        self.learning_rate = learning_rate

        # standart deviation args
        self.sigma = es_config.setdefault('sigma_init', 0.1)
        self.sigma_init = es_config.setdefault('sigma_init', 0.1)
        self.sigma_decay = es_config.setdefault('sigma_decay', 0.999)
        self.sigma_limit = es_config.setdefault('sigma_limit', 0.01)

        # learning rate args
        self.learning_rate = es_config.setdefault('learning_rate', 0.01)
        self.learning_rate_decay = es_config.setdefault('learning_rate_decay', 0.9999)
        self.learning_rate_limit = es_config.setdefault('learning_rate_limit', 0.001)

        # population args
        self.popsize = es_config.setdefault('popsize', 16)
        self.rank_fitness= es_config.setdefault('rank_fitness', True)
        self.forget_best= es_config.setdefault('forget_best', True)

        # train args
        self.antithetic = es_config.setdefault('antithetic', False)
        self.weight_decay = es_config.setdefault('weight_decay', 0.01)

        if self.rank_fitness:
            self.forget_best = True # always forget the best one if we rank

        # discrete or continus
        if self.discrete:
            self.logits_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers)
            self.mean_net = None
            self.logstd = None
        else:
            self.logits_net = None
            self.mean_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers)
            self.logstd = nn.Parameter(
                    -0.5 * torch.ones(self.act_dim, dtype=torch.float32))

        # initialize mu
        self.mu = self.get_flat_params()
        self.best_solution = self.get_flat_params()
        self.best_reward = None

        # calculate parameter dimension
        self.num_params = np.sum([param.numel() for param in self.parameters()])

        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def rms_stdev(self):
        return np.sqrt(np.mean(self.sigma*self.sigma))

    @contextmanager
    def eval(self, seed: int, policy_idx: int=0):
        rng = np.random.RandomState(seed)
        epsilon = rng.randn(self.num_params)
        self.epsilon = epsilon
        self.seed = seed
        self.policy_idx = policy_idx
        self._set_from_flat_params(self, ptu.from_numpy(self.mu))

        if self.antithetic and \
                policy_idx >= int(self.popsize / 2):
            epsilon = -epsilon

        epsilon = ptu.from_numpy(epsilon)
        # add epsilon to original params
        self._add_from_flat_params(self, epsilon * self.sigma)
        yield
        # sub epsilon to original params
        self._sub_from_flat_params(self, epsilon * self.sigma)

    def update(self, reward_list: List):
        '''
            update/train this policy with different algos
        '''
        epsilons = np.vstack([rng.randn(self.num_params) for rng in self.rngs])

        solutions = self.mu.reshape(1, self.num_params) + epsilons * self.sigma

        rewards = np.array(reward_list)
        self.best_reward_raw = np.sort(rewards)[-1]

        if self.rank_fitness:
            rewards = compute_centered_ranks(rewards)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, solutions)
            rewards += l2_decay

        idx = rewards.argsort()[::-1]

        cur_best_reward = rewards[idx[0]]
        cur_best_solution = solutions[idx[0]]

        if self.best_reward is None:
            self.best_reward = cur_best_reward
            self.best_solution = cur_best_solution

        else:
            if self.forget_best or cur_best_reward > self.best_reward:
                self.best_reward = cur_best_reward
                self.best_solution = cur_best_solution

        # standardize the rewards
        normalized_rewards = (rewards - np.mean(rewards)) / np.std(rewards)

        grad = 1./ (self.popsize * self.sigma) * (epsilons.T @ normalized_rewards).sum()

        # self.optimizer.stepsize = self.learning_rate
        # update_ratio = self.optimizer.update(-grad)
        mu = np.copy(self.mu)
        mu += self.learning_rate * grad
        self.mu = np.copy(mu)
        update_ratio = np.mean(self.learning_rate * grad)


        train_log = {}
        train_log['Best Solution'] = self.best_solution
        if self.rank_fitness:
            train_log['Best Reward'] = self.best_reward_raw
        else:
            train_log['Best Reward'] = self.best_reward
        train_log['Mu'] = self.mu
        train_log['Sigma'] = self.sigma
        train_log['Learning Rate'] = self.learning_rate
        train_log['Update Ratio'] = update_ratio

        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

        return train_log

    @property
    def param_dim(self):
        return self._param_dim

    def get_flat_params(self) -> np.ndarray:
        flat_params = torch.cat([param.flatten() for param in self.parameters()])
        return ptu.to_numpy(flat_params)

    def set_rngs(self, seeds: List):
        self.rngs = [np.random.RandomState(seed) for seed in seeds]

    def set_mu(self, mu: torch.Tensor):
        self._set_from_flat_params(self, mu)

    def _set_from_flat_params(
        self,
        module: nn.Module,
        flat_param: torch.Tensor
    ) -> nn.Module:

        pre_idx = 0
        for param in module.parameters():
            flat_size = np.prod(param.shape)
            param.data.copy_(
                flat_param[pre_idx:pre_idx + flat_size].view(param.size())
            )
            pre_idx += flat_size

    def _add_from_flat_params(
        self,
        module: nn.Module,
        flat_param: torch.Tensor
    ) -> nn.Module:

        pre_idx = 0
        for param in module.parameters():
            flat_size = np.prod(param.shape)
            param.data.add_(
                flat_param[pre_idx:pre_idx + flat_size].view(param.size())
            )
            pre_idx += flat_size

    def _sub_from_flat_params(
        self,
        module: nn.Module,
        flat_param: torch.Tensor
    ) -> nn.Module:

        pre_idx = 0
        for param in module.parameters():
            flat_size = np.prod(param.shape)
            param.data.sub_(
                flat_param[pre_idx:pre_idx + flat_size].view(param.size())
            )
            pre_idx += flat_size

