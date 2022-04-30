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
            self.logstd = nn.Parameter(
                    torch.zeros(self.act_dim, dtype=torch.float32, device=ptu.device))
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

    # def set_state(self, state: dict) -> None:

    #     # Set optimizer vars first.
    #     optimizer_vars = state.get("_optimizer_variables", None)
    #     if optimizer_vars:
    #         assert len(optimizer_vars) == len(self._optimizers)
    #         for o, s in zip(self._optimizers, optimizer_vars):
    #             optim_state_dict = ptu.convert_to_torch_tensor(s, device=self.device)
    #             o.load_state_dict(optim_state_dict)
    #     # Set exploration's state.

    #     if hasattr(self, "exploration") and "_exploration_state" in state:
    #         self.exploration.set_state(state=state["_exploration_state"])

    #     # Then the Policy's (NN) weights.
    #     self.set_weights(state['weights'])

    # def set_weights(self, state_dict):
    #     weights = ptu.convert_to_torch_tensor(state_dict, device=ptu.device)
    #     print(weights)
    #     return self.load_state_dict(weights)

    # def get_state(self):
    #     state = self.get_weights()
    #     state["_optimizer_variables"] = []
    #     for i, o in enumerate(self._optimizers):
    #         optim_state_dict = ptu.convert_to_numpy(o.state_dict())
    #         state["_optimizer_variables"].append(optim_state_dict)
    #     # Add exploration state.
    #     if hasattr(self, "exploration"):
    #         self.exploration.set_state(state=state["_exploration_state"])
    #     return state

    # def get_weights(self):
        # return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}

    def set_state(self, state_dict):
        self.load_state_dict(state_dict)

    def get_state(self):
        # return self.state_dict()
        # return {k: v.cpu().detach().numpy() for k, v in self.state_dict().items()}
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
