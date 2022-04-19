import os
import time

from typing import Dict
from reRLs.infrastructure.rl_trainer import Traj_Trainer
from reRLs.agents.pg_agent import PGAgent

class PG_Trainer():

    def __init__(self, config: Dict):

        #####################
        ## SET AGENT CONFIGS
        #####################

        # NN args
        neural_network_args = {
            'layers': config['layers'],
            'learning_rate': config['learning_rate'],
        }

        # adv args
        estimate_advantage_args = {
            'gamma': config['gamma'],
            'standardize_advantages': not(config['dont_standardize_advantages']),
            'reward_to_go': config['reward_to_go'],
            'use_baseline': config['use_baseline'],
            'gae_lambda': config['gae_lambda']
        }

        # update rules, batch_size, buffer_size and other agent args
        train_args = {
            'buffer_size': config['buffer_size'],
            'batch_size': config.setdefault('batch_size', config['itr_size'])
        }

        agent_config = {** neural_network_args, **estimate_advantage_args, **train_args}

        # logger args
        logger_config = {
            'exp_prefix': config['exp_prefix'],
            'seed': config['seed'],
            'exp_id': config['exp_id'],
            'snapshot_mode': config['snapshot_mode']
        }

        self.config = config
        self.config['agent_class'] = PGAgent
        self.config['agent_config'] = agent_config
        self.config['logger_config'] = logger_config

        ################
        ## RL TRAINER
        ################

        self.rl_trainer =  Traj_Trainer(self.config)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_itr = self.config['n_itr'],
            collect_policy = self.rl_trainer.agent.policy,
            eval_policy = self.rl_trainer.agent.policy
        )

def get_parser():

    import argparse
    parser = argparse.ArgumentParser()

    # exp & logger args
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)

    # logger args
    parser.add_argument('--exp_prefix', type=str, default='PG_Cartpole-v1')
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="last")

    # train args
    parser.add_argument('--n_itr', '-n', type=int, default=10)
    parser.add_argument('--itr_size', type=int, default=1000) #steps collected per train iteration
    parser.add_argument('--num_trajectories_eval', '-nte', type=int, default=5) #steps collected per eval iteration
    parser.add_argument('--buffer_size', type=int, default=1000000) #steps collected per train iteration
    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--num_agent_train_steps_per_itr', type=int, default=1)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--tabular_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--num_envs', type=int, default=1)

    # rl common args
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--action_noise_std', type=float, default=0)

    # adv args
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--use_baseline', action='store_true')
    parser.add_argument('--gae_lambda', type=float, default=None)
    parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true')

    # nn args
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=[400,300])

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    # convert to dictionary
    config = vars(args)

    ################
    # RUN TRAINING #
    ################

    trainer = PG_Trainer(config)
    trainer.run_training_loop()

if __name__ == '__main__':
    main()
