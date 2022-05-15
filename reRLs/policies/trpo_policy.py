import abc
import numpy as np
import torch
import itertools
import warnings

from typing import Dict,Union,List, Any, Callable
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions
from torch.distributions import kl_divergence
from reRLs.policies.base_policy import GaussianPolicy
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils import utils

class GaussianPolicyTRPO(GaussianPolicy):

    def __init__(self, obs_dim, act_dim, layers, **kwargs):

        self.delta = kwargs.pop('max_kl')
        self.backtrack_coeff = kwargs.pop('backtrack_coeff')
        self.max_backtracks = kwargs.pop('max_backtracks')
        self.num_critic_update_steps_per_train = kwargs.pop('num_critic_update_steps_per_train')

        super().__init__(obs_dim, act_dim, layers, **kwargs)
        self.baseline_loss = nn.MSELoss()
        # adjusts Hessian-vector product calculation for numerical stability
        self.damping = 0.1

    def _get_log_prob(self, obss, acts):

        ''' get log prob of given state-action paires'''

        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        act_dist = self.forward(obss)
        return ptu.to_numpy(act_dist.log_prob(acts))

    def _get_act_dist(self, obss):
        ''' get action distribution(detached) of given state-action paires'''
        obss = ptu.from_numpy(obss)
        act_dist = self.forward(obss)
        return act_dist

    def update(self, obss, acts, log_pi_old, act_dist_old, advs, q_values=None):
        '''
            Update the policy using trpo surrogate object
        '''

        obss = ptu.from_numpy(obss)
        acts = ptu.from_numpy(acts)
        log_pi_old = ptu.from_numpy(log_pi_old)
        advs = ptu.from_numpy(advs)

        # direction: calculate vallia policy gradient
        act_dist = self.forward(obss)
        log_pi = act_dist.log_prob(acts)
        entropy = act_dist.entropy().mean()

        ratio = torch.exp(log_pi - log_pi_old)
        surr_obj = torch.mul(ratio, advs)
        pi_loss = -surr_obj.mean()

        # weighted_pg = torch.mul(log_pi, advs)
        # pi_loss = -weighted_pg.mean()

        policy = self._get_policy()

        flat_grads = self._get_flat_grad(
            pi_loss, policy, retain_graph=True
        ).detach()

        # direction: calculate natural gradient

        # kl_divergence return same value regradless distribution order
        kl_loss = kl_divergence(act_dist_old, act_dist).mean()

        # calculate first order gradient of kl_loss with respect to policy params
        flat_grads_kl = self._get_flat_grad(
            kl_loss, policy, create_graph=True
        )
        assert flat_grads_kl.mean() < 1e-8, "check flat_grad_kl == 0"

        Hx = lambda x: self._matrix_vector_product(flat_grads_kl, x, policy)
        search_direction = -self._conjugate_gradients(
            Ax=Hx, b=flat_grads
        )

        # stepsize: calculate max stepsize constrained by kl bound
        step_size = torch.sqrt(
            2 * self.delta /
            search_direction.dot(Hx(search_direction))
        )

        # step
        with torch.no_grad():
            flat_params = torch.cat(
                [param.data.view(-1) for param in policy.parameters()]
            )
            for i in range(self.max_backtracks):
                new_flat_params = flat_params + step_size * search_direction
                self._set_from_flat_params(policy, new_flat_params)
                # calculate kl and if in bound, loss actually down
                act_dist_new = self.forward(obss)
                log_pi_new = act_dist_new.log_prob(acts)
                ratio_new = torch.exp(log_pi_new - log_pi_old)
                surr_obj_new = torch.mul(ratio_new, advs)
                pi_loss_new = -surr_obj_new.mean()
                kl = kl_divergence(act_dist_old, act_dist_new).mean()

                # check kl if in bound and loss actually down
                if kl < self.delta and pi_loss_new < pi_loss:
                    if i > 0:
                        warnings.warn(f'Accepting new params at step %d of line search.{i}')
                    break
                elif i < self.max_backtracks - 1:
                    step_size = step_size * self.backtrack_coeff
                else:
                    self._set_from_flat_params(policy, new_flat_params)
                    step_size = torch.zeros_like(step_size)
                    warnings.warn(f'Line search failed!')

            backtrack_iters = i
            improve = pi_loss - pi_loss_new

        if self.use_baseline:

            ## standardize the q_values to have a mean of zero and a standard deviation of one
            ## HINT: there is a `standardize` function in `infrastructure.utils`
            mean_q, std_q = np.mean(q_values), np.std(q_values)
            targets = utils.standardize(q_values, mean_q, std_q)
            targets = ptu.from_numpy(targets)

            for _ in range(self.num_critic_update_steps_per_train):
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
        train_log['Training loss'] = ptu.to_numpy(pi_loss)
        train_log['Entropy'] = ptu.to_numpy(entropy)
        train_log['KL Divergence'] = ptu.to_numpy(kl)
        train_log['Search Direction'] = ptu.to_numpy(search_direction.mean())
        train_log['Step Size'] = ptu.to_numpy(step_size.mean())
        train_log['BackTrack Itr'] = backtrack_iters
        train_log['Improve'] = ptu.to_numpy(improve)

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

    def _conjugate_gradients(
        self,
        Ax: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        nsteps: int=10,
        residual_to_end: float=1e-8
    ) -> torch.Tensor:

        '''
        The idea and pseudo code of conjugate gradient see:
        https://jonathan-hui.medium.com/rl-conjugate-gradient-5a644459137a
        https://zhuanlan.zhihu.com/p/491587841

        The python code of conjugate gradient see:
        https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py
        https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/npg.py
        '''
        # In trpo(npg), b is the vanilla policy gradient g

        # initial solution x0
        x = torch.zeros_like(b)
        # Note: should be 'r0 = b - Ax0', but x0=0
        # Change if doing warm start
        r = b.clone()
        p = r.clone()

        rdotr = r.dot(r)
        for _ in range(nsteps):
            _Ap = Ax(p)
            alpha = rdotr / (p.dot(_Ap))
            x += alpha * p
            r -= alpha * _Ap
            rdotr_new = r.dot(r)
            beta = rdotr_new / rdotr
            p = r + beta * p
            rdotr = rdotr_new
            if rdotr < residual_to_end:
                break
        return x

    def _matrix_vector_product(
        self,
        flat_grad: torch.Tensor,
        x: torch.Tensor,
        module: nn.Module
    ) -> torch.Tensor:
        '''
        compute Hessian Matrix `H_loss` products a given vector `x`

        Since it's `difficult` for pytorch to directly calculate
        Hessian Matrix 'H' with respect to theta and then get 'H*x'.
        We need to first calcualte the product 'flat_grad * x' and
        then get the gradient of 'flat_grad * x', which is equal to 'H*x'.

        Actually, pytorch now support calculate the Hessian Martrix. see
        https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html
        '''

        # calculate matrix prod 'flat_grad * x'
        flat_grad_prod_x = (flat_grad * x).sum() # Try mean?

        # calculate gradient of the previous product which is equal to
        # `H_loss * x`
        flat_grad_grad = self._get_flat_grad(flat_grad_prod_x, module, retain_graph=True)
        return flat_grad_grad + x * self.damping

    def _get_flat_grad(
        self,
        loss: torch.Tensor,
        module: nn.Module,
        **kwargs: Any
    ) -> torch.Tensor:
        '''
        copied from https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/npg.py

        get flatten grad vector of a model given the loss value.

        **kwargs can be any useful args for calculate the grad
        using Pytorch.
        '''
        grads = torch.autograd.grad(loss, module.parameters(), **kwargs)
        return torch.cat([grad.flatten() for grad in grads])

    def _set_from_flat_params(
        self,
        module: nn.Module,
        flat_params: torch.Tensor
    ) -> nn.Module:

        # assert flat_params.requires_grad is True, "The parameters must requires grad"
        pre_idx = 0
        for param in module.parameters():
            flat_size = np.prod(param.shape)
            param.data.copy_(
                flat_params[pre_idx:pre_idx + flat_size].view(param.size())
            )
            pre_idx += flat_size

    def _get_policy(self):

        if self.mean_net is not None:
            return self.mean_net
        else:
            return self.logits_net
