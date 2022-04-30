import gym
import envpool
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from gym.wrappers.normalize import NormalizeObservation

# def make_envs(env_name, num_envs, seed, start_idx=0):
def make_envs(env_name, num_envs, seed, env_type='gym'):
    ''' Helper function to make AsyncVectorEnv '''
    def make_gym_env():
        def fn():
            env = gym.make(env_name)
            env.reset(seed=seed)
            env.action_space.seed(seed)
            return env
        return fn

    def make_env_pool():
        def fn():
            env = envpool.make(env_name, env_type='gym', num_envs=num_envs, seed=seed)
            return env
        return fn

    if num_envs > 1: return make_env_pool()()
    else: return make_gym_env()()

        # vec_env = AsyncVectorEnv(env_fns=[make_env(start_idx+rank) for rank in range(num_envs)])
        # vec_env = SyncVectorEnv(env_fns=[make_env(start_idx+rank) for rank in range(num_envs)])

        # dummy_env = make_env(start_idx)()
        # vec_env.spec = dummy_env.spec
        # vec_env.env = dummy_env
        # dummy_env.close()
        # return vec_env

def get_max_episode_steps(env):

    if hasattr(env, 'spec') and hasattr(env.spec, 'max_episode_steps'):
        return env.spec.max_episode_steps

    elif hasattr(env, 'config'):
        return env.config['max_episode_steps']

    elif hasattr(env, '_max_episode_steps'):
        return env._max_episode_steps

    else:
        raise ValueError(f"Not found the max episode steps of given env {str(env)}")
