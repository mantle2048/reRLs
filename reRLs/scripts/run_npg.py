import os
import time

from typing import Dict
from reRLs.infrastructure.rl_trainer import RL_Trainer
from reRLs.scripts.run_trpo import get_parser, TRPO_Trainer

def main():

    parser = get_parser()
    arg_list =  [
        '--save_params',
        '--use_baseline',
        '--reward_to_go',
        '--backtrack_coeff',
        '1.0',
        '--max_backtracks',
        '1',
    ]
    args = parser.parse_args(args=arg_list)

    # convert to dictionary
    config = vars(args)

    ################
    # RUN TRAINING #
    ################

    trainer = TRPO_Trainer(config)
    trainer.run_training_loop()

if __name__ == '__main__':
    main()
