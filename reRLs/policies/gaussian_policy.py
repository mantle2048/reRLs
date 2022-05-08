import abc
import numpy as np
import torch
import itertools

from typing import List
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions
from reRLs.policies.base_policy import BasePolicy
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils import utils


class GaussianPolicy(BasePolicy, nn.Module, abc.ABC):

    def __init__(
        self,
        obs_dim,
        act_dim,
        layers=[64,64],
        discrete=False,
        learning_rate=1e-3,
        training=True,
        use_baseline=True,
        entropy_coeff=0.01,
        grad_clip=40.0,
        **kwargs
        ):
        super().__init__(**kwargs)

        # init params
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.layers = layers
        self.discrete = discrete
        self.learning_rate = learning_rate
        self.training = training
        self.use_baseline = use_baseline
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip

        # discrete or continus
        if self.discrete:
            self.logits_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers)
            self.logits_net.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(
                params = self.logits_net.parameters(),
                lr = self.learning_rate)
        else:
            self.logits_net = None
            self.mean_net = ptu.build_mlp(input_size=self.obs_dim,
                                            output_size=self.act_dim,
                                            layers=self.layers)
            self.logstd = -0.5 * nn.Parameter(
                    torch.ones(self.act_dim, dtype=torch.float32, device=ptu.device))
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                params = itertools.chain(self.mean_net.parameters(),[self.logstd]),
                lr = self.learning_rate)


        # init baseline
        if self.use_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.obs_dim,
                output_size=1,
                layers=self.layers
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )

        else:
            self.baseline = None

        self.apply(ptu.init_weights)

        # do last policy layer scaling, this will make initial actions have (close to)
        # 0 mean and std, and will help boost performances,
        # see https://arxiv.org/abs/2006.05990, Fig.24 for details
        ptu.scale_last_layer(self.logits_net if self.logits_net else self.mean_net)


    def save(self, filepath=None):
        torch.save(self.state_dict(), filepath)

    def get_action(self, obs: np.ndarray, deterministic=False) -> np.ndarray:
        '''
            query the policy with observation(s) to get selected action(s)
        '''
        if len(obs.shape) == 1:
            obs = obs[None]

        obs = ptu.from_numpy(obs.astype(np.float32))

        act_dist = self.forward(obs)

        if deterministic and not self.discrete:
            act = act_dist.loc
        else:
            act = act_dist.sample()

        if self.discrete and act.shape != ():
            act = act.squeeze()

        return ptu.to_numpy(act)

    def update(self, obss, acts, **kwargs):
         '''
             update/train this policy with different algos
         '''
         raise NotImplementedError

    def forward(self, obs: torch.Tensor):
        '''
        This function defines the forward pass of the network.
        You can return anything you want, but you should be able to differentiate
        through it. For example, you can return a torch.FloatTensor. You can also
        return more flexible objects, such as a
        `torch.distributions.Distribution` object. It's up to you!
        '''
        if self.discrete:
            logits_na = self.logits_net(obs)
            act_dist = distributions.Categorical(logits = logits_na)

        else:
            mean_na = self.mean_net(obs)
            std_na = torch.exp(self.logstd)
            act_dist = distributions.MultivariateNormal(loc=mean_na, scale_tril=torch.diag(std_na))
            # helpful: difference between multivariatenormal and normal sample/batch/event shapes:
            # https://bochang.me/blog/posts/pytorch-distributions/
            # https://ericmjl.github.io/blog/2019/5/29/reasoning-about-shapes-and-probability-distributions/

        return act_dist

    def set_state(self, state_dict):
        self.load_state_dict(state_dict)

    def get_state(self):
        return {k: v.cpu().detach() for k, v in self.state_dict().items()}

class GaussianPolicyPG(GaussianPolicy):

    def __init__(self, obs_dim, act_dim, layers, **kwargs):

        super().__init__(obs_dim, act_dim, layers, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, obss, acts, advs, q_values=None):
        '''
            Update the policy using policy gradient
        '''

        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        advs = ptu.from_numpy(advs)

        # compute the loss that should be optimized when training with policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
            # is the expectation over collected trajectories of:
            # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
            # by the `forward` method
        # HINT3: don't forget that `optimizer.step()` MINIMIZES a loss

        act_dist = self.forward(obss)
        log_pi = act_dist.log_prob(acts)
        entropy = act_dist.entropy().mean()
        weighted_pg = torch.mul(log_pi, advs)
        loss = -torch.sum(weighted_pg) - self.entropy_coeff * entropy

        # optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_baseline:

            ## standardize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `standardize` function in `infrastructure.utils`
            mean_q, std_q = np.mean(q_values), np.std(q_values)
            targets = utils.standardize(q_values, mean_q, std_q)
            targets = ptu.from_numpy(targets)

            ## use the `forward` method of `self.baseline` to get baseline predictions
            baseline_preds = self.baseline(obss).flatten()

            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_preds.shape == targets.shape

            ## compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            ## HINT: use `F.mse_loss`
            baseline_loss = self.baseline_loss(baseline_preds, targets)

            # optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {}
        train_log['Training loss'] = ptu.to_numpy(loss)
        if self.use_baseline:
            train_log['Baseline loss'] = ptu.to_numpy(baseline_loss)
        if self.entropy_coeff:
            train_log['Entropy'] = ptu.to_numpy(entropy)
        return train_log


    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]

class GaussianPolicyPPO(GaussianPolicy):

    def __init__(self, obs_dim, act_dim, layers, **kwargs):

        self.epsilon = kwargs.pop('epsilon')
        super().__init__(obs_dim, act_dim, layers, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def _log_prob_from_dist(self, obss, acts):
        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        acts_dist = self.forward(obss)
        return ptu.to_numpy(acts_dist.log_prob(acts))

    def update(self, obss, acts, log_pi_old, advs, q_values=None):
        '''
            Update the policy using ppo-clip surrogate object
        '''

        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        log_pi_old = ptu.from_numpy(log_pi_old)
        advs = ptu.from_numpy(advs)


        act_dist = self.forward(obss)
        log_pi = act_dist.log_prob(acts)
        entropy = act_dist.entropy().mean()

        ratio = torch.exp(log_pi - log_pi_old)
        surr1 = ratio * advs
        surr2 = ratio.clamp(
            1.0-self.epsilon, 1.0+self.epsilon
        ) * advs
        surrogate_obj = torch.min(surr1, surr2)
        loss = -torch.mean(surrogate_obj) - self.entropy_coeff * entropy

        # Userful extral info
        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        log_ratio = log_pi - log_pi_old
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipped = ratio.gt(1+self.epsilon) | ratio.lt(1-self.epsilon)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean()

        # optimize `loss` using `self.optimizer`
        # HINT: remember to `zero_grad` first
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_baseline:

            ## standardize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `standardize` function in `infrastructure.utils`
            mean_q, std_q = np.mean(q_values), np.std(q_values)
            targets = utils.standardize(q_values, mean_q, std_q)
            targets = ptu.from_numpy(targets)

            ## use the `forward` method of `self.baseline` to get baseline predictions
            baseline_preds = self.baseline(obss).flatten()

            ## avoid any subtle broadcasting bugs that can arise when dealing with arrays of shape
            ## [ N ] versus shape [ N x 1 ]
            ## HINT: you can use `squeeze` on torch tensors to remove dimensions of size 1
            assert baseline_preds.shape == targets.shape

            ## compute the loss that should be optimized for training the baseline MLP (`self.baseline`)
            ## HINT: use `F.mse_loss`
            baseline_loss = self.baseline_loss(baseline_preds, targets)

            # optimize `baseline_loss` using `self.baseline_optimizer`
            # HINT: remember to `zero_grad` first
            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {}
        train_log['Training loss'] = ptu.to_numpy(loss)
        train_log['Entropy'] = ptu.to_numpy(entropy)
        train_log['KL Divergence'] = ptu.to_numpy(approx_kl)
        train_log['Clip Frac'] = ptu.to_numpy(clipfrac)

        if self.use_baseline:
            train_log['Baseline loss'] = ptu.to_numpy(baseline_loss)
        return train_log

    def run_baseline_prediction(self, obs):
        """
            Helper function that converts `obs` to a tensor,
            calls the forward method of the baseline MLP,
            and returns a np array

            Input: `obs`: np.ndarray of size [N, 1]
            Output: np.ndarray of size [N]

        """
        obs = ptu.from_numpy(obs)
        predictions = self.baseline(obs)
        return ptu.to_numpy(predictions)[:, 0]
