import gym
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from gym.wrappers.normalize import NormalizeObservation

def make_envs(env_name, num_envs, seed, start_idx=0, mode='async'):
    ''' Helper function to make AsyncVectorEnv '''
    def make_env(rank):
        def fn():
            env = gym.make(env_name)
            env.reset(seed=seed+rank)
            env.action_space.seed(seed+rank)
            return env
        return fn

    if num_envs > 1:
        if mode == 'async':
            vec_env = AsyncVectorEnv(env_fns=[make_env(start_idx+rank) for rank in range(num_envs)])
        elif mode == 'sync':
            vec_env = SyncVectorEnv(env_fns=[make_env(start_idx+rank) for rank in range(num_envs)])
        else:
            raise ValueError(f"Invialid mode: {mode}")

        dummy_env = make_env(start_idx)()
        vec_env.spec = dummy_env.spec
        vec_env.env = dummy_env
        dummy_env.close()
        return vec_env
    else:
        return make_env(start_idx)()

def get_env_spec_dir(env):
    attrs = []
    names = []
    assert env.spec, 'Env has no spec'
    for name in dir(env.spec):
        if name[0:2] == '__': continue
        if name[0] == '_': continue
        names.append(name)
        attrs.append((getattr(env.spec, name)))
    return dict(zip(names, attrs))
