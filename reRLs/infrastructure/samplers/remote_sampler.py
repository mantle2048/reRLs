import os
import pickle
from collections import OrderedDict
from copy import deepcopy

import ray
import torch
import numpy as np

from .base_sampler import BaseSampler
from .simple_sampler import rollout

from reRLs.infrastructure.utils.utils import Path, get_pathlength
from reRLs.infrastructure.utils import pytorch_util as ptu

class RemoteSampler(BaseSampler):
    def __init__(
        self,
        env,
        trainer_config,
    ):

        super().__init__(env, trainer_config)

        self._trainer_config = trainer_config
        self._num_workers = trainer_config.get('num_workers', 5)
        self._worker_set = None

        self._initialized = False

    def _create_remote_workers(self, env, policy):

        env_pkl = pickle.dumps(env)

        remote_policy = deepcopy(policy)
        remote_policy.cpu()
        policy_pkl = pickle.dumps(remote_policy)
        self._worker_set = WorkerSet(
            env_pkl,
            policy_pkl,
            self._num_workers,
            self._trainer_config
        )

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Block until the env and policy is ready
        initialized = self._worker_set.is_initialized()
        assert initialized, "A fewer Remote Workers are not correctly initialized !"

        self._initialized = True

    def sample(self, policy, timeout=10):
        if not self._initialized:
            self._create_remote_workers(env=self._env, policy=policy)

        # policy_weights = policy.get_weights()
        policy_weights = policy.get_state()
        self._worker_set.sync_weights(policy_weights)

        remote_paths = ray.get(
            [worker.sample.remote(self._render) for worker in self._worker_set.remote_workers()],
            timeout=timeout
        )

        return remote_paths

    def sample_trajectories(self, min_timesteps_per_batch, *args, **kwargs):

        paths = []
        timesteps_this_batch = 0
        while timesteps_this_batch < min_timesteps_per_batch:
            # cur_path_length = min(max_path_length, min_timesteps_per_batch - timesteps_this_batch)
            remote_paths = self.sample(*args, **kwargs)
            for path in remote_paths:
                timesteps_this_batch += get_pathlength(path)
                paths.append(path)
                if timesteps_this_batch >= min_timesteps_per_batch:
                    break

        return paths, timesteps_this_batch

    def sample_n_trajectories(self, ntraj, *args, **kwargs):
        paths = [ self.sample(*args, **kwargs) for _ in range(ntraj) ]
        return paths

    def get_diagnostics(self):
        diagnostics = OrderedDict({})

        return diagnostics

class WorkerSet(object):
    """Set of RolloutWorkers with n @ray.remote workers and zero or one local worker."""
    def __init__(
        self,
        env_pkl,
        policy_pkl,
        num_workers,
        trainer_config
    ):

        self._env_pkl = env_pkl
        self._policy_pkl = policy_pkl
        self._num_workers = num_workers
        self._trainer_config_ref = ray.put(trainer_config)

        self._remote_workers = []
        self.add_workers(self._num_workers)

        self._initialized = False

    def is_initialized(self):
        remote_workers_initialized = ray.get([w.is_initialized.remote() for w in self._remote_workers])
        self._initialized = all(remote_workers_initialized)
        return self._initialized

    def add_workers(self, num_workers: int) -> None:
        """Creates and adds a number of remote workers to this worker set.
        Can be called several times on the same WorkerSet to add more
        RolloutWorkers to the set.
        Args:
            num_workers: The number of remote Workers to add to this
                WorkerSet.
        """
        self._remote_workers.extend(
            [
                self._make_worker(worker_idx=idx)
                for idx in range(num_workers)
            ]
        )

    def _make_worker(self, worker_idx):
        return _RemoteWorker.remote(
            self._env_pkl,
            self._policy_pkl,
            worker_idx,
            self._trainer_config_ref
        )

    def remote_workers(self):
        """Returns a list of remote rollout workers."""
        return self._remote_workers

    def sync_weights(self, weights):
        """Syncs model weights from the provided weights to all remote workers.
        Args:
            policies: Optional list of PolicyIDs to sync weights for.
                If None (default), sync weights to/from all policies.
            from_worker: Optional RolloutWorker instance to sync from.
                If None (default), sync from this WorkerSet's local worker.
        """
        # Only sync if we have remote workers or `from_worker` is provided.
        if self.remote_workers() is not None:
            # Put weights only once into object store and use same object
            # ref to synch to all workers
            weights_ref = ray.put(weights)
            # Sync to all remote workers in this WorkerSet.
            for to_worker in self.remote_workers():
                # to_worker.set_weights.remote(weights_ref)
                to_worker.set_state.remote(weights_ref)


# @ray.remote(num_gpus=0.25)
@ray.remote
class _RemoteWorker(object):
    def __init__(
        self,
        env_pkl,
        policy_pkl,
        worker_idx,
        trainer_config
    ):

        self._env = pickle.loads(env_pkl)
        self._policy = pickle.loads(policy_pkl)
        self._worker_idx = worker_idx

        # ptu.init_gpu(
        #     use_gpu=not trainer_config['no_gpu'],
        #     gpu_id=trainer_config['which_gpu']
        # )

        self._trainer_config = trainer_config
        self._seed = trainer_config['seed'] * 1000 + worker_idx
        self._max_path_length = trainer_config['ep_len'] #TODO: change ep_len -> max_path_length
        self._deterministic = trainer_config['deterministic']

        self._set_env_seed(self._seed)

        if hasattr(self._env, 'initialize'):
            self._env.initialize()

        self._initialized = True

    def _set_env_seed(self, seed):

        if hasattr(self._env, "seed"):
            self._env.seed(seed=seed)
        else:
            self._env.reset(seed=seed)

        self._env.action_space.seed(seed)

    def is_initialized(self):
        return self._initialized

    def set_state(self, policy_weights):
        # policy_weights = ray.get(policy_weights_ref)
        self._policy.set_state(policy_weights)

    def sample(self, render=False):
        path = rollout(
            env=self._env,
            policy=self._policy,
            trainer_config=self._trainer_config,
            render=render
        )
        return path
