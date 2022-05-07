from typing import Dict
from reRLs.infrastructure.utils.utils import *

class ReplayBuffer():

    def __init__(self, max_size = 1000000):

        self.max_size = max_size
        self.paths = []
        self.obs_buf = None
        self.act_buf = None
        self.rew_buf = None
        self.next_obs_buf = None
        self.done_buf = None
        self._initialized = False

    def __len__(self):
        return self.size

    def _init_buffer(self, obs_dim, act_dim):
        """
            Init replay buffer for sample store mode
        """

        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs_buf = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_idx = 0
        self.size = 0
        self._initialized = True


    def add_sample(self, obs, act, rew, next_obs, done):
        """
            store a transition (s, a, r, s', d)
        """
        if not self._initialized:
            self._init_buffer(obs.size, act.size)

        self.obs_buf[self.next_idx] = obs
        self.next_obs_buf[self.next_idx] = next_obs
        self.act_buf[self.next_idx] = act
        self.rew_buf[self.next_idx] = rew
        self.done_buf[self.next_idx] = float(done)

        if self.size < self.max_size:
            self.size += 1

        self.next_idx = (self.next_idx + 1) % self.max_size


    def add_rollouts(self, paths, noised=False):
        """
            store rollouts
        """

        # add new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        # convert new rollouts into their component arrays,
        # and append them onto our arrays
        obss, acts, concated_rews, unconcated_rews, next_obss, dones = \
                convert_listofrollouts(paths)

        if noised:
            obss = add_noise(obss)
            next_obss = add_noise(next_obss)

        for o, a, r, no, d in zip(obss, acts, concated_rews, next_obss, dones):
            self.add_sample(o, a, r, no, d)

    def sample_recent_rollouts(self, num_rollouts=1):

        assert self.paths, "no trajectories saved"
        return self.paths[-num_rollouts:]

    def sample_random_data(self, batch_size) -> Dict:

        assert batch_size <= len(self), "no enough data to sample"
        rand_indices = np.random.randint(len(self), size=batch_size)
        return dict(obs=self.obs_buf[rand_indices],
                    act=self.act_buf[rand_indices],
                    rew=self.rew_buf[rand_indices],
                    next_obs=self.next_obs_buf[rand_indices],
                    done=self.done_buf[rand_indices])

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            assert batch_size <= len(self), "no enough data to sample"
            recent_indices = np.arange(self.next_idx - batch_size, self.next_idx)
            return dict(obs=self.obs_buf[recent_indices],
                        act=self.act_buf[recent_indices],
                        rew=self.rew_buf[recent_indices],
                        next_obs=self.next_obs_buf[recent_indices],
                        done=self.done_buf[recent_indices])

        else:
            assert self.paths, "no trajectories saved, so cannot sample unconcated_rew."
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -= 1
                num_recent_rollouts_to_return += 1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            obss, acts, concated_rews, unconcated_rews, next_obss, dones = \
                    convert_listofrollouts(rollouts_to_return)
            return dict(obs=obss, act=acts, rew=unconcated_rews, next_obs=next_obss, done=dones)

class PPOBuffer():

    def __init__(self):

        self.paths = []
        self.size = 0

    def __len__(self):
        return self.size

    def add_rollouts(self, paths, agent, noised=False):
        """
            store rollouts
            agent: ppo_agent to calculate some useful info of paths
        """
        obss, acts, concated_rews, rews_list, next_obss, dones = \
                convert_listofrollouts(paths)

        # calculate action log probs of old policy
        log_pi = agent.log_prob_from_dist(obss, acts)

        # calculate q values of each (s_t, a_t) point, using rewards [r_1, ..., r_t, ..., r_T]
        q_values = agent.calculate_q_values(rews_list)

        # calculate advantages that correspond to each (s_t, a_t) point
        advs = agent.estimate_advantages(obss, rews_list, q_values, dones)

        # set new rollouts into our list of rollouts
        for path in paths:
            self.paths.append(path)

        if noised:
            obss = add_noise(obss)
            next_obss = add_noise(next_obss)

        self.obs_buf = obss
        self.act_buf = acts
        self.rew_buf = concated_rews
        self.rews_list_buf = rews_list
        self.next_obs_buf = next_obss
        self.done_buf = dones
        self.log_pi_buf = log_pi
        self.adv_buf = advs
        self.q_val_buf = q_values

        self.size = len(concated_rews)

    def sample_random_data(self, batch_size) -> Dict:

        assert batch_size <= len(self), "no enough data to sample"
        rand_indices = np.random.randint(len(self), size=batch_size)
        return dict(obs=self.obs_buf[rand_indices],
                    act=self.act_buf[rand_indices],
                    rew=self.rew_buf[rand_indices],
                    next_obs=self.next_obs_buf[rand_indices],
                    done=self.done_buf[rand_indices],
                    log_pi=self.log_pi_buf[rand_indices],
                    adv=self.adv_buf[rand_indices],
                    q_value=self.q_val_buf[rand_indices])
