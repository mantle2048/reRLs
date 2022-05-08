import reRLs
from reRLs.scripts.run_ppo import get_parser, main, PPO_Trainer
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

def test_ppo(seed=1):
    
    arg_list =  [
        '--env_name',
        'LunarLander-v2',
        '--exp_prefix',
        'PPO_LunarLander-v2',
        '--n_itr',
        '151',
        '--seed',
        f'{seed}',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '1',
        '--itr_size',
        '2000',
        '--batch_size',
        '400',
        '--entropy_coeff',
        '0.01',
        '--gamma',
        '0.995',
        '--gae_lambda',
        '0.98',
        '--num_agent_train_steps_per_itr',
        '5',
        '--save_params',
        '--use_baseline',
        '--num_workers',
        '4',
        '--num_envs',
        '1',
        '-lr',
        '3e-3',
        '--reward_to_go',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PPO_Trainer(config)
    trainer.run_training_loop()


if __name__ == '__main__':
    test_ppo(0)






