# -*- coding: utf-8 -*-
import reRLs
import numpy as np
import torch
import torch.nn as nn
from reRLs.scripts.run_trpo import get_parser, main, TRPO_Trainer
from reRLs.policies.trpo_policy import GaussianPolicyTRPO
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

# # Test torch.autograd and flatten the grad

# +
# model = GaussianPolicyTRPO(10, 3, [64], discrete=False)

def test_flat_grad(verbose=False):
    y = model.mean_net(torch.ones(20, 10)).mean()
    grads = torch.autograd.grad(y, model.mean_net.parameters())
    flat_grad = torch.cat([grad.reshape(-1) for grad in grads])
    if verbose:
        print(model)
        print(y)
        print("==============" * 8)
        print('first_layer_weight_grad: ', grads[0].shape)
        print('first_layer_bias_grad: ', grads[1].shape)
        print("==============" * 8)
        print('second_layer_weight_grad: ', grads[2].shape)
        print('second_layer_bias_grad: ', grads[3].shape)
        print("==============" * 8)
        print('requires grad: ', flat_grad.requires_grad)
        print('flat grad shape: ', flat_grad.shape)
        print("==============" * 8)
    return flat_grad


# -

# # Test set flat_param to a model

def test_set_flat_param():
    flat_grads = test_flat_grad()
    flat_params = torch.cat([param.flatten() for param in model.mean_net.parameters()])
    flat_params += flat_grads
    model._set_from_flat_params(model.mean_net, flat_params)
# test_set_flat_param()


# # Test calculate KL Divergence via Pytorch

from torch.distributions import kl_divergence
import torch.nn.functional as F
import torch.distributions as dis
def test_kl():
    p = dis.Normal(loc=0, scale=1)
    q = dis.Normal(loc=1, scale=1)

    x = p.sample(sample_shape=(1_000_0,))
    log_p = p.log_prob(x)
    log_q = q.log_prob(x)

    log_ratio = log_q - log_p
    ratio = torch.exp(log_ratio)
    approx_kl = ((ratio - 1) - log_ratio).mean()

    print("kl divergence \\approx log(p(x) / q(x))")
    print("==============" * 8)
    print('true_kl:         ', kl_divergence(p, q).mean())
    print('lower_variance kl[q, p]:  ', approx_kl)
    print('lower_variance kl[p, q]:  ', ( ratio * log_ratio - (ratio - 1)).mean())
    print('higher_variance: ', (log_p - log_q).mean())
    print("==============" * 8)
    print("kl divergence loss = p(x) * log(p(x) / q(x))")
    print("==============" * 8)
    # 注意input是q target是p 顺序相反
    print(F.kl_div(input=log_q, target=log_p, log_target=True, reduction='mean'))
    print((torch.exp(log_p) * (log_p - log_q)).mean())


# # Test TRPO

def test_trpo(seed=1):
    
    arg_list =  [
        '--env_name',
        'LunarLander-v2',
        '--exp_prefix',
        'TRPO_LunarLander-v2',
        '--n_itr',
        '1',
        '--seed',
        f'{seed}',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '1',
        '--itr_size',
        '2000',
        '--batch_size',
        '400',
        '--gamma',
        '0.99',
        '--gae_lambda',
        '0.99',
        '--num_agent_train_steps_per_itr',
        '4',
        '--save_params',
        '--use_baseline',
        '--num_workers',
        '5',
        '--num_envs',
        '1',
        '-lr',
        '1e-3',
        '--reward_to_go',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = TRPO_Trainer(config)
    trainer.run_training_loop()


if __name__ == '__main__':
    test_trpo(seed=0)


# # Test NPG

def test_npg(seed=1):
    
    arg_list =  [
        '--env_name',
        'LunarLander-v2',
        '--exp_prefix',
        'NPG_LunarLander-v2',
        '--n_itr',
        '1',
        '--seed',
        f'{seed}',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '1',
        '--itr_size',
        '2000',
        '--batch_size',
        '400',
        '--gamma',
        '0.99',
        '--gae_lambda',
        '0.99',
        '--num_agent_train_steps_per_itr',
        '4',
        '--save_params',
        '--use_baseline',
        '--num_workers',
        '5',
        '--num_envs',
        '1',
        '-lr',
        '1e-3',
        '--reward_to_go',
        '--backtrack_coeff',
        '1.0',
        '--max_backtracks',
        '1',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = TRPO_Trainer(config)
    trainer.run_training_loop()


if __name__ == '__main__':
    test_npg(seed=0)


