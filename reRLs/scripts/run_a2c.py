import os
import time

from typing import Dict
from reRLs.infrastructure.rl_trainer import RL_Trainer
from reRLs.scripts.run_pg import get_parser, PG_Trainer

def main():

    parser = get_parser()
    arg_list =  [
        '--save_params',
        '--reward_to_go',
        '--use_baseline'
    ]
    args = parser.parse_args(args=arg_list)

    # convert to dictionary
    config = vars(args)

    ################
    # RUN TRAINING #
    ################

    trainer = PG_Trainer(config)
    trainer.run_training_loop()

if __name__ == '__main__':
    main()
