from reRLs.infrastructure.rl_trainer import *
from reRLs.infrastructure.utils.gym_util import *
from reRLs.infrastructure.loggers.tabulate import tabulate
# %matplotlib notebook
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

env = make_envs(env_name="HalfCheetah-v3", num_envs=1, seed=0)
print(env.spec.max_episode_steps)

env.model.opt.timestep

env.env.metadata

attrs = []
names = []
for name in dir(env.spec):
    if name[0:2] == '__': continue
    if name[0] == '_': continue
    names.append(name)
    attrs.append((getattr(env.spec, name)))
print(dict(zip(names,attrs)))

obs = env.reset()
print(obs.shape)
env.close()

async_env = make_envs(env_name="HalfCheetah-v3", num_envs=10, seed=0)

obs = async_env.reset()
print(obs.shape)

async_env.spec.max_episode_steps

async_env.metadata

async_env.metadata

async_env.close()


