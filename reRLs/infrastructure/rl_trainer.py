from collections import OrderedDict
import pickle
import os
import sys
import time
import abc
import time
import gym
import torch
import numpy as np

from gym import wrappers
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
from reRLs.infrastructure.samplers.simple_sampler import SimpleSampler

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40


class RL_Trainer(object):

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
            use_gpu=not self.config['no_gpu'],
            gpu_id=self.config['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        self.env = make_envs(
            env_name=self.config['env_name'],
            num_envs=self.config['num_envs'],
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
        if 'model' in dir(self.eval_env):
            self.fps = 1 / self.eval_env.mode.opt.timestep
        elif 'env_wrappers' in self.config:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.eval_env.env.metadata.keys():
            self.fps = self.eval_env.env.metadata['video.frames_per_second']
        elif 'render_fps' in self.eval_env.env.metadata.keys():
            self.fps = self.eval_env.env.metadata['render_fps']
        else:
            self.fps = 10

        #############
        ## AGENT
        #############

        agent_class = self.config['agent_class']
        self.agent = agent_class(self.env, self.config['agent_config'])

        #############
        ## Sampler
        #############
        sampler_class = samplers.get(self.config['num_workers'], self.config['num_envs'])
        self.train_sampler = sampler_class(self.env, self.config)
        self.eval_sampler = SimpleSampler(self.eval_env, self.config)

    ####################################
    ####################################

    def run_training_loop(self, n_itr, collect_policy, eval_policy):
        """
        :param n_itr:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
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

            ## collect trajectories, to be used for training

            train_returns = self.collect_training_trajectories(
                itr, self.config['itr_size'], collect_policy)

            paths, envsteps_this_itr, train_video_paths = train_returns
            self.total_envsteps += envsteps_this_itr

            ## add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            ## train agent (using sampled data from replay buffer)
            train_logs = self.train_agent(itr)

            ## log/save
            if self.logtabular:
                ## perform tabular and video
                self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs)

                if self.config['save_params']:
                    self.logger.save_itr_params(itr, self.agent.policy.get_state())
                    self.logger.save_extra_data(data = self.agent.what_to_save())

        self.env.close()
        self.eval_env.close()
        self.logger.close()

    def collect_training_trajectories(self, itr, itr_size, collect_policy,  load_initial_expertdata=None):

        # if your load_initial_expertdata is None, then you need to collect new trajectories at *every* iteration
        if itr == 0 and load_initial_expertdata is not None:
            import pickle
            with open(load_initial_expertdata, 'rb') as fr:
                loaded_paths = pickle.load(fr)
                return loaded_paths, 0, None

        print("\nCollecting trajectories to be used for training...")
        paths, env_steps_this_itr = \
                self.train_sampler.sample_trajectories(itr_size, collect_policy)

        train_video_path = None
        if self.logvideo:
            print("\nCollecting train rollouts to be used for saving videos...")
            with self.train_sampler.render():
                train_video_path = self.train_sampler.sample_n_trajectories(MAX_NVIDEO, collect_policy)

        return paths, env_steps_this_itr, train_video_path

    ####################################
    ####################################

    def train_agent(self, itr):

        train_logs = []
        for train_step in range(self.config['num_agent_train_steps_per_itr']):
            data_batch = self.agent.sample(self.config.setdefault('batch_size', self.config['itr_size']))
            # if itr == 0 and train_step == 1:
            train_log, continue_training = self.agent.train(data_batch)
            train_logs.append(train_log)
            ## continue_training tag is used for early stop training
            ## when some constraints (like target kl) are not satisfied
            if not continue_training:
                self.logger.log(f"Early stopping at Itr {itr}")
                break

        return train_logs

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        if itr == 0:
            ## log and save config_json
            self.logger.log_variant('config.json', self.config)

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths = self.eval_sampler.sample_n_trajectories(
            self.config.setdefault('num_trajectories_eval', 10),
            eval_policy
        )

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            with self.eval_sampler.render():
                eval_video_paths = self.eval_sampler.sample_n_trajectories(MAX_NVIDEO, eval_policy)

            ## save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(
                train_video_paths, itr, fps=self.fps,
                max_videos_to_save=MAX_NVIDEO, video_title='train_rollouts'
            )
            self.logger.log_paths_as_videos(
                eval_video_paths, itr, fps=self.fps,
                max_videos_to_save=MAX_NVIDEO, video_title='eval_rollouts'
            )

        #######################

        # save eval tabular
        if self.logtabular:
            # returns, for logging
            train_returns = [path["rew"].sum() for path in paths]
            eval_returns = [eval_path["rew"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["rew"]) for path in paths]
            eval_ep_lens = [len(eval_path["rew"]) for eval_path in eval_paths]

            # decide what to log
            self.logger.record_tabular("Itr", itr)

            self.logger.record_tabular_misc_stat("EvalReward", eval_returns)
            self.logger.record_tabular_misc_stat("TrainReward", train_returns)

            self.logger.record_tabular("EvalEpLen", np.mean(eval_ep_lens))
            self.logger.record_tabular("TrainEpLen", np.mean(train_ep_lens))
            self.logger.record_tabular("TotalEnvInteracts", self.total_envsteps)
            self.logger.record_tabular("Time", (time.time() - self.start_time) / 60)
            self.logger.record_dict(last_log)

            self.logger.dump_tabular(with_prefix=True, with_timestamp=False)
