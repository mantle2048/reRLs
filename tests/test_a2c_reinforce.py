import reRLs
from reRLs.scripts.run_pg import get_parser, main, PG_Trainer
# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

# # Create a virtual screen

def test_virtual_screen():

    import pyvirtualdisplay
    _display = pyvirtualdisplay.Display(visible=False,  # use False with Xvfb
                                        size=(1400, 900))
    _ = _display.start()

    print(_.stop())

# # Test Vanilla PG #

def test_vanilla_pg():

    # ### get config
    # add 'args=[]' in ( ) for useage of jupyter notebook
    args = get_parser().parse_args(args=[
        '--n_itr',
        '101',
        '--seed',
        '2',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '10'
    ])
    config = vars(args)
    from pprint import pprint
    pprint(config)

    # ### Run trainer
    trainer = PG_Trainer(config)
    trainer.run_training_loop()

# # Test PG with reward to go

def test_pg_rtg():

    arg_list =  [
        '--exp_prefix',
        'PG-rtg_CartPole-v1',
        '--n_itr',
        '101',
        '--seed',
        '2',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '10',
        '--reward_to_go'
    ]
    args = get_parser().parse_args(args=arg_list)
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()

# # Test PG with reward to go and use_baseline

def test_pg_reg_baseline():

    arg_list =  [
        '--exp_prefix',
        'PG-rtg-baseline_CartPole-v1',
        '--n_itr',
        '101',
        '--seed',
        '2',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '10',
        '--reward_to_go',
        '--use_baseline'
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()

# # Test PG with reward to go and use_baseline and gae

def test_pg_rtg_baseline_gae():
    arg_list =  [
        '--exp_prefix',
        'PG-rtg-baseline-gae_CartPole-v1',
        '--n_itr',
        '101',
        '--seed',
        '3',
        '--video_log_freq',
        '-1',
        '--tabular_log_freq',
        '1',
        '--reward_to_go',
        '--use_baseline',
        '--gamma',
        '0.98',
        '--gae_lambda',
        '0.97',
        '--save_params',
        '-lr',
        '3e-4'
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()

# # Test pygame video record

def test_pygame_video_record():

    arg_list =  [
        '--env_name',
        'Pendulum-v1',
        '--exp_prefix',
        'PG-rtg-baseline-gae_Pendulum-v1',
        '--n_itr',
        '101',
        '--seed',
        '3',
        '--video_log_freq',
        '10',
        '--tabular_log_freq',
        '1',
        '--reward_to_go',
        '--use_baseline',
        '--gamma',
        '0.98',
        '--gae_lambda',
        '0.97',
        '--save_params',
        '-lr',
        '3e-4'
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()

# # Test Mujoco video record

def test_mujoco_video_record():

    arg_list =  [
        '--env_name',
        'HalfCheetah-v3',
        '--ep_len',
        '150',
        '--exp_prefix',
        'PG-rtg-baseline-gae_HalfCheetah-v3',
        '--n_itr',
        '11',
        '--seed',
        '3',
        '--video_log_freq',
        '10',
        '--tabular_log_freq',
        '1',
        '--reward_to_go',
        '--use_baseline',
        '--gamma',
        '0.98',
        '--gae_lambda',
        '0.97',
        '--save_params',
        '-lr',
        '3e-4'
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()


# # Test PG

def test_reinforce(seed=1):
    env_name = 'LunarLander-v2'
    arg_list =  [
        '--env_name',
        env_name,
        '--exp_prefix',
        f'REINFORCE_{env_name}',
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
        '--gamma',
        '0.995',
        '--save_params',
        '--reward_to_go',
        '--num_workers',
        '4',
        '--num_envs',
        '1',
        '-lr',
        '3e-3',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()


# # Test A2C 

def test_a2c(seed=1):
    env_name = 'LunarLander-v2'
    arg_list =  [
        '--env_name',
        env_name,
        '--exp_prefix',
        f'A2C_{env_name}',
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
        '--gamma',
        '0.995',
        '--gae_lambda',
        '0.98',
        '--save_params',
        '--reward_to_go',
        '--use_baseline',
        '--entropy_coeff',
        '0.01',
        '--num_workers',
        '4',
        '--num_envs',
        '1',
        '-lr',
        '3e-3',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = PG_Trainer(config)
    trainer.run_training_loop()


if __name__ == '__main__':
    test_reinforce(2)



