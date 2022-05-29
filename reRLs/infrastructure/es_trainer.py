import time
import gym
import torch
import numpy as np
from typing import Dict
from collections import OrderedDict
from pyvirtualdisplay import Display
from gym.wrappers.normalize import NormalizeObservation
from reRLs.infrastructure.utils import utils
from reRLs.infrastructure.loggers import setup_logger
from reRLs.infrastructure.utils import pytorch_util as ptu
from reRLs.infrastructure.wrappers import ActionNoiseWrapper
from reRLs.infrastructure.utils.gym_util import make_envs, get_max_episode_steps
from reRLs.infrastructure import samplers
from reRLs.infrastructure.samplers.simple_sampler import rollout
from reRLs.infrastructure.samplers.simple_sampler import SimpleSampler

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40

class ES_Trainer(object):

    def __init__(self, config: Dict):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.config = config
        self.logger = setup_logger(**self.config['logger_config'])
        self.virtual_disp = Display(visible=False, size=(1400,900))
        self.virtual_disp.start()

        # Set random seed
        seed = self.config.setdefault('seed', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=False
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = make_envs(
            env_name=self.config['env_name'],
            num_envs=1,
            seed=self.config['seed']
        )

        self.eval_env = make_envs(
            env_name=self.config['env_name'],
            num_envs=1,
            seed=self.config['seed'] + 100
        )


        # Add normalize observation wrapper
        if self.config.setdefault('obs_norm', False):
            self.env = NormalizeObservation(self.env)
            self.eval_env = NormalizeObservation(self.eval_env)

        # Add noise wrapper
        if self.config.setdefault('action_noise_std', 0) > 0:
            self.env = ActionNoiseWrapper(self.env, self.config['seed'], self.config['action_noise_std'])

        # Maximum length for episodes
        # self.ep_len = self.config.setdefault('ep_len', self.env.spec.max_episode_steps)
        self.config['ep_len'] = self.config['ep_len'] or get_max_episode_steps(self.env)
        self.config['agent_config']['ep_len'] = self.config['ep_len']
        self.ep_len = self.config['ep_len']
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.config['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.eval_env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.eval_env.observation_space.shape) > 2

        self.config['agent_config']['discrete'] = discrete

        # Observation and action sizes
        obs_dim = self.eval_env.observation_space.shape if img else self.eval_env.observation_space.shape[0]
        act_dim = self.eval_env.action_space.n if discrete else self.eval_env.action_space.shape[0]
        self.config['agent_config']['obs_dim'] = obs_dim
        self.config['agent_config']['act_dim'] = act_dim

        # simulation timestep, will be used for video saving
        self.fps=30

        #############
        ## AGENT
        #############

        agent_class = self.config['agent_class']
        self.agent = agent_class(self.env, self.config['agent_config'])
        self.eval_sampler = SimpleSampler(self.eval_env, self.config)


    ####################################
    ####################################

    def run_training_loop(self, n_itr, eval_policy):
        """
        :param n_itr:  number of iterations
        :param eval_policy: used for eval the performance of mu
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_itr):

            ## decide if videos should be rendered/logged at this iteration
            if self.config['video_log_freq'] != -1 \
                    and itr % self.config['video_log_freq'] == 0:
                self.logvideo = True
            else:
                self.logvideo = False

            ## decide if tabular should be logged
            if self.config['tabular_log_freq'] != -1 \
                    and itr % self.config['tabular_log_freq'] == 0:
                self.logtabular = True
            else:
                self.logtabular = False

            ## solutions can be params or seeds
            solutions, env_steps_this_itr = self.ask_agent()
            ## train agent (using sampled data from replay buffer)
            self.total_envsteps += env_steps_this_itr

            train_logs = self.tell_agent(solutions)

            ## log/save
            if self.logtabular:
                ## perform tabular and video
                self.perform_logging(itr, eval_policy, train_logs)

                if self.config['save_params']:
                    self.logger.save_itr_params(itr, self.agent.policy.get_state())

        self.env.close()
        self.eval_env.close()
        self.logger.close()

    def ask_agent(self):

        # sample a batch random seeds and eval es_policy to obtain episode reward
        reward_list, env_steps_this_itr = self.agent.ask()
        return reward_list, env_steps_this_itr

    def tell_agent(self, solutions):

        train_logs = []
        for train_step in range(self.config['num_agent_train_steps_per_itr']):
            train_log = self.agent.tell(solutions)
            train_logs.append(train_log)

        return train_logs


    ####################################
    ####################################

    def perform_logging(self, itr, eval_policy, all_logs):

        last_log = all_logs[-1]

        #######################

        if itr == 0:
            ## log and save config_json
            self.logger.log_variant('config.json', self.config)

        #######################

        print("\nCollecting data for eval...")
        eval_paths = self.eval_sampler.sample_n_trajectories(
            self.config.setdefault('num_trajectories_eval', 10),
            eval_policy
        )

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo:
            print('\nCollecting video rollouts eval')
            with self.eval_sampler.render():
                best_mu = ptu.from_numpy(eval_policy.best_solution)
                eval_policy.set_mu(best_mu)
                best_eval_video_paths = self.eval_sampler.sample_n_trajectories(1, eval_policy)

                cur_mu = ptu.from_numpy(eval_policy.mu)
                eval_policy.set_mu(cur_mu)
                mu_eval_video_paths = self.eval_sampler.sample_n_trajectories(1, eval_policy)
                eval_video_paths = best_eval_video_paths + mu_eval_video_paths
                self.logger.log("==========")
                self.logger.log(str(best_eval_video_paths[0]['rew'].sum()))
                self.logger.log(str(mu_eval_video_paths[0]['rew'].sum()))
                self.logger.log("==========")

            ## save train/eval videos
            print('\nSaving eval rollouts as videos...')
            self.logger.log_paths_as_videos(
                eval_video_paths, itr, fps=self.fps,
                max_videos_to_save=MAX_NVIDEO, video_title='eval_rollouts'
            )

        #######################

        # save eval tabular
        if self.logtabular:
            # returns, for logging
            mu = last_log.pop("Mu")
            best_solution = last_log.pop("Best Solution")

            mu_returns = [eval_path["rew"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            mu_ep_lens = [len(eval_path["rew"]) for eval_path in eval_paths]

            # decide what to log
            self.logger.record_tabular("Itr", itr)

            self.logger.record_tabular_misc_stat("MuReward", np.array(mu_returns))

            self.logger.record_tabular("MuEpLen", np.mean(mu_ep_lens))
            self.logger.record_tabular("TotalEnvInteracts", self.total_envsteps)
            self.logger.record_tabular("Time", (time.time() - self.start_time) / 60)
            self.logger.record_dict(last_log)

            self.logger.log_distribution(mu, "Mu", itr)
            self.logger.log_distribution(best_solution, "Best Solution", itr)

            self.logger.dump_tabular(with_prefix=True, with_timestamp=False)
