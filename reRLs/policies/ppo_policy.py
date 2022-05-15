import abc
import numpy as np
import torch
import itertools

from typing import List
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions
from torch.distributions import kl_divergence
from reRLs.policies.base_policy import GaussianPolicy
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils import utils

class GaussianPolicyPPO(GaussianPolicy):

    def __init__(self, obs_dim, act_dim, layers, **kwargs):

        self.epsilon = kwargs.pop('epsilon')
        super().__init__(obs_dim, act_dim, layers, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def _get_log_prob(self, obss, acts):
        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        act_dist = self.forward(obss)
        return ptu.to_numpy(act_dist.log_prob(acts))

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
