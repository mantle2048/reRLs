# +
import envpool
import numpy as np
import reRLs
from reRLs.infrastructure.wrappers import ActionNoiseWrapper
from reRLs.infrastructure.wrappers import PreprocessStepWrapper

# %matplotlib notebook
# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2
# -

# make asynchronous
num_envs = 4
batch_size = 4
env = envpool.make("Ant-v3", env_type="gym", num_envs=num_envs, seed=0)
action_num = env.action_space.shape[0]
env.config

obs = env.reset(env_id=np.array([0]))
obs = env.reset(env_id=np.array([1]))
obs = env.reset(env_id=np.array([2]))
obs = env.reset(env_id=np.array([3]))
print(obs.shape)

print(env)

dir(env)

for i in range(2):
    # obs, rew, done, info = env.recv()
    # env_id = info['env_id']
    action = np.stack([env.action_space.sample() for _ in range(batch_size)])
    next_obs, rew, done, info = env.step(action)
    print(rew)
    # print(env._cur_obs.sum() == next_obs.sum())
    
    # print(action)
    # env.send(action, env_id)

# +
# dir(env)
# 
# env.action_space.sample()
# 
# env = ActionNoiseWrapper(env, 0, 0.1)
# 
# env.config
# 
# hasattr(env.config, 'max_episode_steps')
# 
# hasattr(env.spec.gen_config(), 'max_episode_steps')
# 
# env.render(mode='rgb_array')
# 
# env


# -


