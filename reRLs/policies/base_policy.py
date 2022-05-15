import abc
import numpy as np
import torch
import itertools

from typing import List
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils import utils

class BasePolicy(abc.ABC):

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError

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
        entropy_coeff=0.00,
        grad_clip=None,
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
                    -0.5 * torch.ones(self.act_dim, dtype=torch.float32, device=ptu.device))

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
