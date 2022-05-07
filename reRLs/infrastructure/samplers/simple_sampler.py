from collections import defaultdict

import numpy as np
from .base_sampler import BaseSampler
from reRLs.infrastructure.utils.utils import Path, get_pathlength

class SimpleSampler(BaseSampler):
    def __init__(self, env, trainer_config):

        super().__init__(env, trainer_config)
        self._cur_obs = self._env.reset(seed=self._seed)

    def sample(self, policy):
        obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []
        steps = 0
        while True:
            if self._render:
                if 'rgb_array' in self._render_mode:
                    if hasattr(self._env, 'sim'):
                        image_obss.append(self._env.sim.render(camera_name='track', height=500, width=500)[::-1])
                    else:
                        image_obss.append(self._env.render(mode='rgb_array'))
                if 'human' in self._render_mode:
                    self._env.render(mode=self._render_mode)
                    time.sleep(self._env.model.opt.timestep)

            obs = self._cur_obs
            obss.append(obs)
            act = policy.get_action(obs, self._deterministic)
            if len(act.shape) > 1:
                act = act[0]
            acts.append(act)

            next_obs, rew, done, _ = self._env.step(act)

            rews.append(rew)
            next_obss.append(next_obs)

            steps += 1

            rollout_done = done or steps >= self._max_path_length
            terminals.append(rollout_done)
            self._cur_obs = next_obs

            if rollout_done:
                self._cur_obs = self._env.reset()
                break

        return Path(obss, image_obss, acts, rews, next_obss, terminals)

    def sample_trajectories(self, min_timesteps_per_batch, *args, **kwargs):

        paths = []
        timesteps_this_batch = 0
        while timesteps_this_batch < min_timesteps_per_batch:
            # cur_path_length = min(max_path_length, min_timesteps_per_batch - timesteps_this_batch)
            path = self.sample(*args, **kwargs)
            timesteps_this_batch += get_pathlength(path)
            paths.append(path)

        return paths, timesteps_this_batch

    def sample_n_trajectories(self, ntraj, *args, **kwargs):
        paths = [ self.sample(*args, **kwargs) for _ in range(ntraj) ]
        return paths

    def get_diagnostics(self):
        diagnostics = super().get_diagnostics()
        diagnostics.update({})

        return diagnostics

def rollout(
    env,
    policy,
    trainer_config,
    sampler_class=SimpleSampler,
    render=False,
):
    sampler = sampler_class(env, trainer_config)
    if render:
        with sampler.render():
            path = sampler.sample(policy)
    else:
        path = sampler.sample(policy)
    return path

def rollouts(n_trajs, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_trajs)]
    return paths

