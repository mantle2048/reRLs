import torch
from typing import Dict
from reRLs.infrastructure.utils.utils import *

class BatchBuffer():

    def __init__(self):

        self.paths = {}
        self.size = 0

    def __len__(self):
        return self.size

    def add_rollouts(self, paths, paths_info, noised=False):
        """
            store rollouts
            agent: agent to calculate some useful info of paths
        """
        raise NotImplementedError

    def sample_random_data(self, batch_size) -> Dict:
        """
            sample random data
        """

        assert batch_size <= len(self), "no enough data to sample"
        rand_indices = np.random.randint(len(self), size=batch_size)
        batch_data = {}
        for key, value in self.paths.items():
            batch_data[key] = value[rand_indices]
        return batch_data

class PPOBuffer(BatchBuffer):

    def __init__(self):
        super().__init__()

    def add_rollouts(self, paths, agent, noised=False):
        """
            store rollouts
            agent: ppo_agent to calculate some useful info of paths
        """

        obss, acts, concated_rews, rews_list, next_obss, dones = \
                convert_listofrollouts(paths)


        # calculate action log probs of old policy
        log_pi = agent.get_log_prob(obss, acts)

        # calculate q values of each (s_t, a_t) point, using rewards [r_1, ..., r_t, ..., r_T]
        q_values = agent.calculate_q_values(rews_list)

        # calculate advantages that correspond to each (s_t, a_t) point
        advs = agent.estimate_advantages(obss, rews_list, q_values, dones)

        if noised:
            obss = add_noise(obss)
            next_obss = add_noise(next_obss)

        # store new paths and new paths_info
        self.paths['obs'] = obss
        self.paths['act'] = acts
        self.paths['rew'] = concated_rews
        self.paths['next_obs'] = next_obss
        self.paths['done'] = dones
        self.paths['log_pi'] = log_pi
        self.paths['adv'] = advs
        self.paths['q_value'] = q_values

        self.size = len(concated_rews)

class TRPOBuffer(BatchBuffer):
    def __init__(self):
        super().__init__()
        self.dist = None

    def add_rollouts(self, paths, agent, noised=False):
        """
            store rollouts
            agent: npg or trpo agent to calculate some useful info of paths
        """

        obss, acts, concated_rews, rews_list, next_obss, dones = \
                convert_listofrollouts(paths)

        # calculate action log probs of old policy
        log_pi = agent.get_log_prob(obss, acts)

        # calculate q values of each (s_t, a_t) point, using rewards [r_1, ..., r_t, ..., r_T]
        q_values = agent.calculate_q_values(rews_list)

        # calculate advantages that correspond to each (s_t, a_t) point
        advs = agent.estimate_advantages(obss, rews_list, q_values, dones)

        if noised:
            obss = add_noise(obss)
            next_obss = add_noise(next_obss)

        # store new paths and new paths_info
        self.paths['obs'] = obss
        self.paths['act'] = acts
        self.paths['rew'] = concated_rews
        self.paths['next_obs'] = next_obss
        self.paths['done'] = dones
        self.paths['log_pi'] = log_pi
        self.paths['adv'] = advs
        self.paths['q_value'] = q_values

        self.size = len(concated_rews)

    def sample_random_data(self, batch_size, agent) -> Dict:
        """
            sample random data
        """

        assert batch_size <= len(self), "no enough data to sample"
        rand_indices = np.random.randint(len(self), size=batch_size)
        batch_data = {}
        for key, value in self.paths.items():
            batch_data[key] = value[rand_indices]
        obss = batch_data['obs']

        with torch.no_grad():
            batch_data['act_dist_old'] = agent.get_act_dist(obss)
        return batch_data
