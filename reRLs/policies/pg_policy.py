import abc
import numpy as np
import torch
import itertools

from typing import List
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions
from reRLs.policies.base_policy import GaussianPolicy
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils import utils

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
