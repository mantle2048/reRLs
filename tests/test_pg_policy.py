import numpy as np
from pprint import pprint as pp

# # Test generalized advantage estimator (GAE)

# +
baselines = np.ones(100)

baselines = np.append(baselines, [0])

rews = np.ones(100)
advs = np.zeros_like(baselines)
gamma = 1.
gae_lambda = 1.
dones = np.stack([0,0,0,0,0,0,0,0,0,1] * 10)
deltas = rews + gamma * baselines[1:] * (1 - dones) - baselines[:-1]

pp(deltas)

for i in reversed(range(100)):
    advs[i] = gamma * gae_lambda * advs[i+1] * (1 - dones[i]) + deltas[i]

advs[:-1]
