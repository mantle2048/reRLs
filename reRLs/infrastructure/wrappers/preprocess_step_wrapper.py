import gym
import numpy as np


class PreprocessStepWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._cur_obs = None

    def reset(self):
        self._cur_obs, _, _, info = self.env.recv()
        self._cur_env_id = info['env_id']
        return self._cur_obs

    def step(self, act):
        self.env.send(act, self._cur_env_id)
        next_obs, rew, done, info = self.env.recv()
        self._cur_env_id = info['env_id']
        self._cur_obs = next_obs

        return next_obs, rew, done, info
