import numpy as np
from scipy.signal import lfilter
from typing import Dict,Union,List

from .base_agent import BaseAgent
from reRLs.policies import GaussianPolicyPG
from reRLs.infrastructure.buffers import ReplayBuffer
from reRLs.infrastructure.utils import utils
from reRLs.infrastructure.utils import pytorch_util as ptu

class PGAgent(BaseAgent):

    def __init__(self, env, agent_config: Dict):
        super().__init__()

        # init params
        self.env = env
        self.agent_config = agent_config
        self.gamma = self.agent_config.setdefault('gamma', 0.99)
        self.standardize_advantages = \
                self.agent_config.setdefault('standardize_advantages', True)
        self.use_baseline = self.agent_config.setdefault('use_baseline', True)
        self.reward_to_go = self.agent_config.setdefault('reward_to_go', True)
        self.gae_lambda = self.agent_config.setdefault('gae_lambda', 0.99)

        self.buffer_size = self.agent_config.setdefault('buffer_size', 1000000)

        self._create_policy_and_buffer()


    def _create_policy_and_buffer(self):

        # policy
        self.policy = GaussianPolicyPG(
            obs_dim = self.agent_config['obs_dim'],
            act_dim = self.agent_config['act_dim'],
            layers  = self.agent_config['layers'],
            discrete = self.agent_config['discrete'],
            learning_rate=self.agent_config['learning_rate'],
            use_baseline=self.agent_config['use_baseline'],
            entropy_coeff=self.agent_config['entropy_coeff'],
            grad_clip=self.agent_config['grad_clip']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def train(self, data_batch) -> Dict:

        """
            Training a PG agent refers to updating its policy using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """
        obss, acts, rews_list, next_obss, dones = \
                data_batch['obs'], data_batch['act'], data_batch['rew'], \
                data_batch['next_obs'], data_batch['done']

        # step 1: calculate q values of each (s_t, a_t) point, using rewards [r_1, ..., r_t, ..., r_T]
        q_values = self.calculate_q_values(rews_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advs = self.estimate_advantages(obss, rews_list, q_values, dones)

        # step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
        ## HINT: `train_log` should be returned by the actor update method
        train_log = self.policy.update(obss, acts, advs, q_values)

        continue_training = True

        return train_log, continue_training

    def calculate_q_values(self, rews_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        if not self.reward_to_go:

            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
            # In other words: value of (s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            q_values = np.concatenate([self._discounted_return(self.gamma, rews) for rews in rews_list])

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:

            # For each point (s_t, a_t), associate its value as being the discounted sum of rewards over the full trajectory
            # In other words: value of (s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            q_values = np.concatenate([self._discounted_cumsum(self.gamma, rews) for rews in rews_list])

        return q_values

    def estimate_advantages(self, obss, rews_list, q_values, dones):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """
        # Estimate the advantage when use_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        if self.use_baseline:
            baselines_standardized = self.policy.run_baseline_prediction(obss)
            ## ensure that the baseline and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert baselines_standardized.ndim == q_values.ndim
            ## baseline was trained with standardized q_values, so ensure that the predictions
            ## have the same mean and standard deviation as the current batch of q_values
            baselines =  \
                    utils.de_standardize(baselines_standardized, np.mean(q_values), np.std(q_values))

            if self.gae_lambda is not None:
                ### append a dummy T+1 value for simpler recursive calculation
                baselines = np.append(baselines, [0])

                ### combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ### create empty numpy array to populate with GAE advantage
                ### estimates, with dummy T+1 value for simpler recursive calculation
                advs = np.zeros_like(baselines)

                deltas = rews + self.gamma * baselines[1:] * (1 - dones) - baselines[:-1]

                for i in reversed(range(obss.shape[0])):
                    advs[i] = self.gamma * self.gae_lambda * advs[i+1] * (1 - dones[i]) + deltas[i]

                ### remove dummy advantage
                advs = advs[:-1]

            else:
                ### compute advantage estimates using q_values and baselines
                advs = q_values - baselines

        # Else, just set the advantage to [Q]
        else:
            advs = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            ## HINT: there is a `standardize` function in `infrastructure.utils`
            advs = utils.standardize(advs, np.mean(advs), np.std(advs))

        return advs

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew = False)

    def what_to_save(self):
        pass

    def _discounted_return(self, discount, rewards: List) -> np.ndarray:
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """
        discounted_returns = np.ones_like(rewards) * self._discounted_cumsum(discount, rewards)[0]
        return discounted_returns

    def _discounted_cumsum(self, discount, rewards: List) -> np.ndarray:
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        discounted_cursums = lfilter([1], [1, -discount], rewards[::-1], axis=0)[::-1]
        return discounted_cursums
