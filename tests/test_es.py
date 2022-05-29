from reRLs.agents.es_agent import *
from reRLs.policies.es_policy import *
from reRLs.scripts.run_es import get_parser, OpenES_Trainer
import numpy as np
import reRLs
import gym
from matplotlib import pyplot as plt
# %matplotlib notebook
# %reload_ext autoreload
# %autoreload 2

# # Test compute ranks

x = np.random.randn(1000)
compute_ranks(x).sum() == x.argsort().sum()

# # Test centered ranks

# x = np.array([[70,6,8],[5,3,4],[0,1,2]])
x = np.array([1,2,3,4,5])
compute_centered_ranks(x)

# # Test Adam lr decay

stepsize = 0.01
beta2 = 0.999
beta1 = 0.99
a = []
for t in range(100):
    a.append(stepsize * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t))
fig = plt.figure(figsize=(4,3))
ax = fig.add_subplot(111)
ax.plot(np.arange(100), a)
fig.show()

# # Test policy n_dim

es_config = dict(
    learning_rate=0.01,         
    antithetic=True,           
    weight_decay=0.01,         
    rank_fitness=True,        
    forget_best=True
)
pi = GaussianPolicyES(obs_dim=8, act_dim=4, layers=[64,64], es_config=es_config)
print(pi.num_params)

# # Test policy eval contextmanager

# +
seed = 0

with torch.no_grad():
    param_sum = 0
    for param in pi.parameters():
        param_sum += param.sum()
    print('before:', end='')
    print(param_sum)
    
with pi.eval(seed), torch.no_grad():
    param_sum = 0
    for param in pi.parameters():
        param_sum += param.sum()
    print('eval:', end='')
    print(param_sum)
    
with torch.no_grad():
    param_sum = 0
    for param in pi.parameters():
        param_sum += param.sum()
    print('after:', end='')
    print(param_sum)
# -
# # Test policy update

reward_list = list(range(10))
seeds = list(range(10))
pi.set_rngs(seeds)
pi.update(reward_list)


pi.update(reward_list)

pi.update(reward_list)

# # Test es_agent

env = gym.make("LunarLander-v2")
env.reset(seed=0)
agent_config = dict(
    obs_dim=8,
    act_dim=4,
    layers=[64, 64],
    discrete=True,
    learning_rate=0.01,
    ep_len=1000,
    popsize=4,
    deterministic=False,
    antithetic=True,           
    weight_decay=0.01,         
    seed=0,
    rank_fitness=True,        
    forget_best=True
)
agent = OpenESAgent(env, agent_config)

seeds = agent.sample_random_seeds()
print(seeds)

ask_return = agent.ask()
print(agent.ask())

reward_list, _ = ask_return
train_log = agent.tell(reward_list)
print(train_log)

train_log = agent.tell(reward_list)
print(train_log)


# # Test Trainer

def test_es(seed=1):
    
    arg_list =  [
        '--env_name',
        'LunarLander-v2',
        '--exp_prefix',
        'ES_LunarLander-v2',
        '--popsize',
        '10',
        '--n_itr',
        '1001',
        '--seed',
        f'{seed}',
        '--video_log_freq',
        '50',
        '--tabular_log_freq',
        '1',
        '--num_agent_train_steps_per_itr',
        '10',
        '--learning_rate',
        '0.0003',
        '--sigma_init',
        '0.01',
        '--sigma_limit',
        '0.001',
        '--save_params',
        # '--antithetic',
        # '--rank_fitness',
    ]
    args = get_parser().parse_args(args=arg_list) # add 'args=[]' in ( ) for useage of jupyter notebook
    config = vars(args)
    from pprint import pprint
    pprint(config)

    trainer = OpenES_Trainer(config)
    trainer.run_training_loop()


if __name__ == '__main__':
    test_es(0)










