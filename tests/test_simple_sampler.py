# +
import pickle
import unittest
import pytest
import gym
import reRLs
import numpy as np
from reRLs.policies.gaussian_policy import GaussianPolicy
from reRLs.infrastructure import samplers
from reRLs.infrastructure.replay_buffer import ReplayBuffer
from reRLs.infrastructure.utils.utils import get_pathlength, write_gif

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline
# -

@pytest.mark.skip(reason="SimpleSampler is currently broken.")
class SimplerSamplerTest(unittest.TestCase):
    #def __init__(self):
    def setUp(self):
        self.env = gym.make("Walker2d-v3")
        self.policy = GaussianPolicy(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0])
        trainer_config= {
            'seed': 0,
            'ep_len': 1000,
            'deterministic': False,
        }
        self.simple_sampler = samplers.get(1)(self.env, trainer_config)
        self.replay_buffer = ReplayBuffer() 
        
    def test_sample(self):
        path = self.simple_sampler.sample(self.policy)
        print(get_pathlength(path))
        print('Sampled path rew :', path['rew'])
        
    def test_sample_trajectories(self):
        paths, env_steps_this_itr = self.simple_sampler.sample_trajectories(1000, self.policy)
        self.replay_buffer.add_rollouts(paths)
        batch = self.replay_buffer.sample_recent_data(1000, concat_rew=False)
        self.assertEqual(np.concatenate(batch['rew']).sum(), np.concatenate([path['rew'] for path in paths]).sum())
        
    def test_sample_n_trajectories(self, n_traj=10):
        paths = self.simple_sampler.sample_n_trajectories(n_traj, self.policy)
        self.assertEqual(len(paths), n_traj)
        
    def test_render(self):
        self.assertEqual(self.simple_sampler._render, False)
        with self.simple_sampler.render():
            self.assertEqual(self.simple_sampler._render, True)
        self.assertEqual(self.simple_sampler._render, False)
    
    def test_rener_sample(self):
        with self.simple_sampler.render():
            render_sample = self.simple_sampler.sample(self.policy)
        print(env.env.metadata)
        fps = self.env.env.metadata['render_fps']
        write_gif('test_render_sample', render_sample['image_obs'], fps=fps)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


