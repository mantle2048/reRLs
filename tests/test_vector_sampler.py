# +
import pickle
import unittest
import pytest
import gym
import reRLs
import envpool
import numpy as np
from reRLs.policies.gaussian_policy import GaussianPolicy
from reRLs.infrastructure import samplers
from reRLs.infrastructure.replay_buffer import ReplayBuffer
from reRLs.infrastructure.utils.utils import get_pathlength, write_gif
from reRLs.infrastructure.utils.gym_util import make_envs

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline
# -

@pytest.mark.skip(reason="SimpleSampler is currently broken.")
class VectorSamplerTest(unittest.TestCase):
    #def __init__(self):
    def setUp(self):
        self.env = envpool.make("Ant-v3", env_type="gym", num_envs=8, seed=0)
        self.policy = GaussianPolicy(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0])
        trainer_config= {
            'seed': 0,
            'ep_len': 1000,
            'deterministic': False,
        }
        self.vector_sampler = samplers.get(num_workers=1, num_envs=8)(self.env, trainer_config)
        self.replay_buffer = ReplayBuffer() 
        
    def test_sample(self):
        vector_env_path = self.vector_sampler.sample(self.policy)
        for path in vector_env_path:
            print(get_pathlength(path))
            print('Sampled path rew :', path['rew'])
        
    def test_sample_trajectories(self):
        paths, env_steps_this_itr = self.vector_sampler.sample_trajectories(1000, self.policy)
        self.replay_buffer.add_rollouts(paths)
        batch = self.replay_buffer.sample_recent_data(env_steps_this_itr, concat_rew=False)
        self.assertEqual(np.concatenate(batch['rew']).sum(), np.concatenate([path['rew'] for path in paths]).sum())
        
    def test_sample_n_trajectories(self, n_traj=10):
        paths = self.vector_sampler.sample_n_trajectories(n_traj, self.policy)
        self.assertEqual(len(paths), n_traj)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)




