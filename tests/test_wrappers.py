# %matplotlib notebook
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
from reRLs.infrastructure.wrappers import ActionNoiseWrapper
from reRLs.infrastructure.rl_trainer import make_envs

env = make_envs(env_name="HalfCheetah-v3", num_envs=1, seed=0)
env = ActionNoiseWrapper(env, seed=1, std=0.1)
print(env)

env = make_envs(env_name="HalfCheetah-v3", num_envs=10, seed=0)
env = ActionNoiseWrapper(env, seed=1, std=0.1)
print(env)
