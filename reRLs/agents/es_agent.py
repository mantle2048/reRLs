import ray
import torch
import pickle
import random
import numpy as np
from typing import Dict, List
from copy import deepcopy
# from reRLs.infrastructure.samplers import rollout
from reRLs.policies.es_policy import GaussianPolicyES
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.utils.utils import Path, get_pathlength


class Seeder:
    def __init__(self, init_seed=0):
        np.random.seed(init_seed)
        self.limit = np.int32(2*32-1)

    def next_seed(self):
        result = np.random.randint(self.limit)
        return result
    def next_batch(self, batch_size):
        result = np.random.randint(self.limit, size=batch_size).tolist()
        return result

class OpenESAgent:

    def __init__(self, env, agent_config: Dict):

        # init params
        self.env = env
        self.agent_config = agent_config

        # popsize
        self.popsize = self.agent_config.setdefault('popsize', 10)
        assert self.popsize % 2 == 0, 'population size should be even'

        # policy
        self.policy = GaussianPolicyES(
            obs_dim = self.agent_config['obs_dim'],
            act_dim = self.agent_config['act_dim'],
            layers  = self.agent_config['layers'],
            discrete = self.agent_config['discrete'],
            es_config = self.agent_config
        )

        # seeder
        self.seeder = Seeder(self.agent_config['seed'])

        # deepcopy env and policy for remote
        self.env_remote = deepcopy(self.env)
        self.policy_remote = deepcopy(self.policy)

        # initialize remote workers
        self.es_workers = WorkerSet(
            self.env_remote,
            self.policy_remote,
            self.popsize,
            self.agent_config
        )

    def sample_random_seeds(self):

        if self.agent_config['antithetic']:
            seeds_half = self.seeder.next_batch(int(self.popsize/2))
            seeds = seeds_half + seeds_half
        else:
            seeds = self.seeder.next_batch(self.popsize)

        return seeds

    def ask(self):

        seeds = self.sample_random_seeds()
        self.seeds = seeds
        # seeds = [0, 1, 2, 3]
        reward_list, env_steps_this_itr = self.es_workers.eval(seeds)
        self.policy.set_rngs(seeds)

        return reward_list, env_steps_this_itr

    def tell(self, reward_list):

        assert (len(reward_list) == self.popsize), "Inconsistent reward size reported."
        self.es_workers.update(reward_list)

        # update local es policy for log training status
        train_log = self.policy.update(reward_list)
        cur_mu = ptu.from_numpy(self.policy.mu)
        self.policy.set_mu(cur_mu)

        return train_log

class WorkerSet(object):
    """Set of RolloutWorkers with n @ray.remote workers and zero or one local worker."""
    def __init__(
        self,
        env,
        policy,
        num_workers,
        trainer_config
    ):

        self._env = env
        self._policy = policy
        self._num_workers = num_workers
        self._trainer_config_ref = ray.put(trainer_config)

        self._remote_workers = []
        self.add_workers(self._num_workers)

    def add_workers(self, num_workers) -> None:
        """Creates and adds a number of remote workers to this worker set.
        Can be called several times on the same WorkerSet to add more
        RolloutWorkers to the set.
        Args:
            num_workers: The number of remote Workers to add to this
                WorkerSet.
        """
        self._remote_workers.extend(
            [
                self._make_worker(
                    env=self._env,
                    policy=self._policy,
                    worker_idx=idx
                )
                for idx in range(num_workers)
            ]
        )

    def _make_worker(self, env, policy, worker_idx):
        return _RemoteWorker.remote(
            env,
            policy,
            worker_idx,
            self._trainer_config_ref
        )

    def remote_workers(self):
        """Returns a list of remote rollout workers."""
        return self._remote_workers

    def torch_random(self):
        returns = ray.get([ \
                worker.torch_random.remote() \
                for worker in self.remote_workers()
            ]
        )
        return returns


    def eval(self, seeds):
        sample_returns = ray.get([ \
                worker.sample.remote(seeds) \
                for worker in self.remote_workers()
            ]
        )
        reward_list = []
        env_steps_this_itr = 0
        for (ep_rew, ep_len) in sample_returns:
            reward_list.append(ep_rew)
            env_steps_this_itr += ep_len
        return reward_list, env_steps_this_itr

    def update(self, reward_list: List):
        for worker in self.remote_workers():
            worker.update.remote(reward_list)

    def get_mus(self):
        mus_return = ray.get([ \
                worker.get_mu.remote() \
                for worker in self.remote_workers()
            ]
        )
        return mus_return

def rollout(env, policy, trainer_config):
    cur_obs = env.reset()
    obss, acts, rews, next_obss, terminals, image_obss = [], [], [], [], [], []
    steps = 0
    while True:

        obs = cur_obs
        obss.append(obs)
        act = policy.get_action(obs, trainer_config['deterministic'])
        if len(act.shape) > 1:
            act = act[0]
        acts.append(act)

        next_obs, rew, done, _ = env.step(act)

        rews.append(rew)
        next_obss.append(next_obs)

        steps += 1

        rollout_done = done or steps >= trainer_config['ep_len']
        terminals.append(rollout_done)
        cur_obs = next_obs

        if rollout_done:
            break

    return Path(obss, image_obss, acts, rews, next_obss, terminals)

def rollouts(n_trajs, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_trajs)]
    return paths

@ray.remote
class _RemoteWorker:

    def  __init__(
        self,
        env,
        policy,
        worker_idx,
        trainer_config
    ):

        self._env = env
        self._policy = policy
        self._worker_idx = worker_idx

        self._trainer_config = trainer_config
        self._seed = trainer_config['seed'] + worker_idx * 1000

        self.set_seed(self._seed)

    def get_mu(self):
        return self._policy.mu.mean()

    def get_best(self):
        return self._policy.best_solution.mean()

    def get_policy(self):
        return self._policy

    def set_seed(self, seed):

        if hasattr(self._env, "seed"):
            self._env.seed(seed=seed)
        else:
            self._env.reset(seed=seed)

        self._env.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(0)

    def sample(self, seeds: List):

        self._policy.set_rngs(seeds)
        seed = seeds[self._worker_idx]

        with self._policy.eval(seed=seed,policy_idx=self._worker_idx):
            paths = rollouts(
                n_trajs=1,
                env=self._env,
                policy=self._policy,
                trainer_config=self._trainer_config,
            )
        ep_rew = np.mean([path["rew"].sum() for path in paths])
        ep_len = np.mean([len(path["rew"]) for path in paths])
        return ep_rew, ep_len

    def update(self, reward_list: List):
        self._policy.update(reward_list)
