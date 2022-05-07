import reRLs
from reRLs.scripts.run_pg import get_parser, main, PG_Trainer
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

def test_a2c(seed=1):
    
    arg_list =  [
        '--env_name',
        'LunarLander-v2',
        '--exp_prefix',
        'A2C_LunarLander-v2',
        '--n_itr',
        '301',
        '--seed',
        f'{seed}',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '1',
        '--itr_size',
        '1000',
        '--batch_size',
        '1000',
        '--gamma',
        '0.99',
        '--save_params',
        '--num_workers',
        '2',
        '--num_envs',
        '1',
        '-lr',
        '3e-3',
        '--gae_lambda',
        '0.99',
        '--use_baseline',
        '-rtg',
        '--use_baseline',
        '--entropy_coeff',
        '0.01',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()


if __name__ == '__main__':
    # test_mujoco_video_record()
    # test_pygame_video_record()
    test_a2c(2)


