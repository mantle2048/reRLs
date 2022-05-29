import os
import time

from typing import Dict
from reRLs.infrastructure.es_trainer import ES_Trainer
from reRLs.agents.es_agent import OpenESAgent

class OpenES_Trainer():

    def __init__(self, config: Dict):

        #####################
        ## SET AGENT CONFIGS
        #####################

        # NN args
        neural_network_args = {
            'layers': config['layers'],
        }

        # train args
        agent_train_args = {
            'deterministic': config['deterministic'],
            'antithetic': config['antithetic'],
            'weight_decay': config['weight_decay'],
            'seed': config['seed'],
        }

        #sigma args
        sigma_args = {
            'sigma_init': config['sigma_init'],
            'sigma_decay': config['sigma_decay'],
            'sigma_limit': config['sigma_limit'],
        }

        # learning rate args
        learning_rate_args = {
            'learning_rate': config['learning_rate'],
            'learning_rate_decay': config['learning_rate_decay'],
            'learning_rate_limit': config['learning_rate_limit'],
        }

        # population args
        population_args = {
            'popsize': config['popsize'],
            'rank_fitness': config['rank_fitness'],
            'forget_best': config['forget_best'],
        }

        agent_config = {
            **neural_network_args,
            **population_args,
            **learning_rate_args,
            **sigma_args,
            **agent_train_args
        }

        # logger args
        logger_config = {
            'exp_prefix': config['exp_prefix'],
            'seed': config['seed'],
            'exp_id': config['exp_id'],
            'snapshot_mode': config['snapshot_mode']
        }

        self.config = config
        self.config['agent_class'] = OpenESAgent
        self.config['agent_config'] = agent_config
        self.config['logger_config'] = logger_config

        ################
        ## RL TRAINER
        ################

        self.es_trainer =  ES_Trainer(self.config)

    def run_training_loop(self):

        self.es_trainer.run_training_loop(
            n_itr = self.config['n_itr'],
            eval_policy = self.es_trainer.agent.policy
        )

def get_parser():

    import argparse
    parser = argparse.ArgumentParser()

    # exp args
    parser.add_argument('--env_name', type=str, default='LunarLander-v2')

    # logger args
    parser.add_argument('--exp_prefix', type=str, default='ES_LunarLander-v2')
    parser.add_argument('--exp_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot_mode', type=str, default="last")

    # train args
    parser.add_argument('--n_itr', '-n', type=int, default=10)
    parser.add_argument('--num_trajectories_eval', '-nte', type=int, default=5) #steps collected per eval iteration
    parser.add_argument('--ep_len', type=int) #students shouldn't change this away from env's default
    parser.add_argument('--num_agent_train_steps_per_itr', type=int, default=1)
    parser.add_argument('--video_log_freq', type=int, default=-1)
    parser.add_argument('--tabular_log_freq', type=int, default=1)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--antithetic', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # wrapper args
    parser.add_argument('--obs_norm', action='store_true')
    parser.add_argument('--action_noise_std', type=float, default=0)

    # nn args
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=[64,64])

    #sigma args
    parser.add_argument('--sigma_init', type=float, default=0.1)
    parser.add_argument('--sigma_decay', type=float, default=0.999)
    parser.add_argument('--sigma_limit', type=float, default=0.01)

    # learning rate args
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--learning_rate_decay', type=float, default=0.9999)
    parser.add_argument('--learning_rate_limit', type=float, default= 0.001)

    # population args
    parser.add_argument('--popsize', type=int, default=10)
    parser.add_argument('--rank_fitness', action='store_true')
    parser.add_argument('--forget_best', action='store_true')

    return parser

def main():

    parser = get_parser()
    args = parser.parse_args()

    # convert to dictionary
    config = vars(args)

    ################
    # RUN TRAINING #
    ################

    trainer = OpenES_Trainer(config)
    trainer.run_training_loop()

if __name__ == '__main__':
    main()
