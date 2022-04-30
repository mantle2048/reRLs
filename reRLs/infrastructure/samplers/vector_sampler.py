import envpool

import numpy as np
from .base_sampler import BaseSampler
from collections import defaultdict
from reRLs.infrastructure.utils.utils import Path, get_pathlength

class VectorSampler(BaseSampler):

    def __init__(self,vector_env, trainer_config):

        super().__init__(vector_env, trainer_config)
        self.vector_env = self._env
        self.num_envs = self.vector_env.config['num_envs']

        self.batch_data = Batch_data(self.num_envs)

        self._observation_space = self.vector_env.observation_space
        self._action_space = self.vector_env.action_space

        self._cur_obs = self.vector_env.reset() # lazily initialized

        assert self.vector_env.config['max_episode_steps'] \
                == self._max_path_length, 'max episode steps not equal'

    def sample(self, policy):

        vector_env_path = [] # may multiple terminal in the same step, so may multiple paths are collected

        while True:
            ## EnvPool doesn't support render now

            obs = self._cur_obs
            act = policy.get_action(obs, self._deterministic)
            assert act.shape[0] == self.num_envs

            next_obs, rew, done, info = self.vector_env.step(act)

            batch = {
                'obs':obs,
                'act':act,
                'rew':rew,
                'next_obs': next_obs,
                'done': done
            }
            self.batch_data.append(batch, info['env_id'])

            self._cur_obs = next_obs

            terminal_env_ids = done.nonzero()[0]

            if terminal_env_ids.size > 0:

                for env_id in terminal_env_ids:
                    path = self.batch_data.poll(env_id)
                    vector_env_path.append(path)
                    self._cur_obs[env_id] = self.vector_env.reset(env_id=np.array([env_id]))
                break

        return vector_env_path

    def sample_trajectories(self, min_timesteps_per_batch, *args, **kwargs):

        paths = []
        timesteps_this_batch = 0
        while timesteps_this_batch < min_timesteps_per_batch:
            # cur_path_length = min(max_path_length, min_timesteps_per_batch - timesteps_this_batch)
            vector_env_path = self.sample(*args, **kwargs)
            for path in vector_env_path:
                timesteps_this_batch += get_pathlength(path)
                paths.append(path)
                if timesteps_this_batch >= min_timesteps_per_batch:
                    break

        return paths, timesteps_this_batch

    def sample_n_trajectories(self, ntraj, *args, **kwargs):
        paths = [ self.sample(*args, **kwargs) for _ in range(ntraj) ]
        return paths


class Batch_data(object):

    def __init__(self, num_envs):

        # self._obss = defaultdict(list)
        # self._acts = defaultdict(list)
        # self._rews = defaultdict(list)
        # self._next_obss = defaultdict(list)
        # self._image_obss = defaultdict(list)
        # self._dones = defaultdict(list)
        # self._infos = defaultdict(list)

        self.num_envs = num_envs
        self._obss = [[] for _ in range(self.num_envs)]
        self._acts = [[] for _ in range(self.num_envs)]
        self._rews = [[] for _ in range(self.num_envs)]
        self._next_obss = [[] for _ in range(self.num_envs)]
        self._dones = [[] for _ in range(self.num_envs)]

        self._image_obss = [[] for _ in range(self.num_envs)]
        self._infos = [[] for _ in range(self.num_envs)]

    def append(self, batch_data, env_ids):
        assert env_ids.all() == np.array(range(self.num_envs)).all(), 'Env ids Error'

        if len(batch_data['obs'].shape) == 1:
            obss = np.hsplit(batch_data['obs'], self.num_envs)
        else:
            obss = np.vsplit(batch_data['obs'], self.num_envs)

        if len(batch_data['act'].shape) == 1:
            acts = np.hsplit(batch_data['act'], self.num_envs)
        else:
            acts = np.vsplit(batch_data['act'], self.num_envs)

        if len(batch_data['next_obs'].shape) == 1:
            next_obss = np.hsplit(batch_data['next_obs'], self.num_envs)
        else:
            next_obss = np.vsplit(batch_data['next_obs'], self.num_envs)

        if len(batch_data['rew'].shape) == 1:
            rews = np.hsplit(batch_data['rew'], self.num_envs)
        else:
            rews = np.vsplit(batch_data['rew'], self.num_envs)

        if len(batch_data['done'].shape) == 1:
            dones = np.hsplit(batch_data['done'], self.num_envs)
        else:
            dones = np.vsplit(batch_data['done'], self.num_envs)

        [x.append(y.squeeze()) for x,y in zip(self._obss, obss)]
        [x.append(y.squeeze()) for x,y in zip(self._acts, acts)]
        [x.append(y.squeeze()) for x,y in zip(self._rews, rews)]
        [x.append(y.squeeze()) for x,y in zip(self._next_obss, rews)]
        [x.append(y.squeeze()) for x,y in zip(self._dones, dones)]

    def poll(self, env_id):
        assert env_id < self.num_envs

        path = Path(
            self._obss[env_id],
            [],
            self._acts[env_id],
            self._rews[env_id],
            self._next_obss[env_id],
            self._dones[env_id]
        )

        self.clear_path(env_id)
        return path

    def clear_path(self, env_id):

        self._obss[env_id] = []
        self._acts[env_id] = []
        self._rews[env_id] = []
        self._next_obss[env_id] = []
        self._dones[env_id] = []

        self._image_obss[env_id] = []
        self._infos[env_id] = []

    def append_image_obss(self, image_obss):
        pass

    def append_info(self, infos):
        for x, y in zip(self._infos, infos):
            x.append(y)










