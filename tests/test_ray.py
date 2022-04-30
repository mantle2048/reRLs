# -*- coding: utf-8 -*-
# +
import ray
import numpy as np
import time
import gym
import pickle

# %reload_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline
# -

ray.init(num_cpus=8, num_gpus=1, ignore_reinit_error=True)


# # test_remote_function

# +
def test_remote_function():
    # 一个常规的python函数
    def regular_function():
        return 1
    
    # 一个 Ray 远程函数
    @ray.remote
    def remote_function():
        return 1
    
    assert regular_function() == 1
    
    object_id = remote_function.remote()
    assert ray.get(object_id) == 1
    
    # These happen serially
    serial_results = []
    for _ in range(4):
        serial_results.append(regular_function())
    print("Serial Outputs:", end=' ')
    print(serial_results)
    
    # These happen in parallel.
    parallel_results = ray.get([remote_function.remote() for _ in range(4)])
    print("Parallel Outputs:", end=' ')
    print(parallel_results)
        
test_remote_function()


# -

# # Test remote class

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0
    def increment(self):
        self.value += 1
        return self.value


def test_remote_class():
    cnt = Counter.remote()
    parallel_results = ray.get([cnt.increment.remote() for _ in range(10)])
    print(parallel_results)
test_remote_class()


# # Test Parameter Server

@ray.remote
class ParameterServer(object):
    def __init__(self, dim):
        # Alternatively, params could be a dictionary mapping keys to arrays.
        self.params = np.zeros(dim)
        
    def get_params(self):
        return self.params
    
    def update_params(self, grad):
        self.params += grad


@ray.remote
def worker(ps):
    '''
    Note that the work function takes a handle to the parameter server as an
    argument, which allows the worker to invoke methods on the parameter
    server actor.
    '''
    for _ in range(100):
        ## Get the latest parameters.
        params_id = ps.get_params.remote() # This method call is non-blocking and returns a future.
        
        params = ray.get(params_id) # This is a blocking call which waits for the task to finish and get the results
        
        ## Compute a gradient update. Here we just make a fake upadte, but in
        ## practice this would use a library like Pytorch and would also take
        ## a batch of data.
        grad = np.ones(10)
        time.sleep(0.2) # This is a fake placeholder for some computation
        
        ## Update the parameters.
        ps.update_params.remote(grad)


def test_param_server():
    ps = ParameterServer.remote(10)
    
    # Start 2 workers.
    for _ in range(2):
        worker.remote(ps)
    obj_id = ps.get_params.remote()
    parallel_results = ray.get(obj_id)
    print(parallel_results)
test_param_server()


class A():
    def __init__(self):
        self.cnt = 1


@ray.remote
class B(A):
    def __init__(self):
        self.cnt = 2


a = B.remote()

a


# +
@ray.remote
class RemoteEnv(object):
    def __init__(self, env_ref, worker_idx):
        self._env = env_ref
        self._seed = worker_idx
        self.reset(worker_idx)
        self._env.action_space.seed(worker_idx)
    
    def env(self):
        return self._env
    
    def reset(self, seed=0):
        return self._env.reset(seed=seed)
    
    def sample(self):
        path_return = 0.
        self._env.reset()
        while True:
            next_obs, rew, done, info = self._env.step(self._env.action_space.sample())
            path_return += rew
            if done:
                break
        return path_return
    
@ray.remote
class RemoteEnv_pkl(object):
    def __init__(self, env_pkl, worker_idx):
        self._env = pickle.loads(env_pkl)
        self._seed = worker_idx
        self.reset(worker_idx)
        self._env.action_space.seed(worker_idx)
    
    def env(self):
        return self._env
    
    def reset(self, seed=0):
        return self._env.reset(seed=seed)
    
    def sample(self):
        path_return = 0.
        self._env.reset()
        while True:
            next_obs, rew, done, info = self._env.step(self._env.action_space.sample())
            path_return += rew
            if done:
                break
        return path_return


# +
def test_same_instance_seed_to_remote():
    env = gym.make("Swimmer-v3")
    env_ref = ray.put(env)
    print(env_ref)
    remote_env = RemoteEnv.remote(env_ref)
    remote_obs_ref = remote_env.reset.remote()
    print(remote_obs_ref)
    remote_obs = ray.get(remote_obs_ref)
    obs = env.reset(seed=0)
    print(obs.sum())
    print(remote_obs.sum())
    
    remote_env2 = RemoteEnv.remote(env_ref)
    
    path_return1_ref = remote_env.sample.remote()
    path_return2_ref = remote_env2.sample.remote()
    path_returns = ray.get([path_return1_ref, path_return2_ref])
    
def test_remote_worker(num_worker=26):
    import time
    from pprint import pprint as pp
    start_time = time.time()
    env = gym.make("Swimmer-v3")
    env_ref = ray.put(env)
    remote_workers = [RemoteEnv.remote(env_ref, idx) for idx in range(num_worker)]
    # path_return_refs, _ = ray.wait([remote_worker.sample.remote() for remote_worker in remote_workers])
    path_return_refs = [remote_worker.sample.remote() for remote_worker in remote_workers]
    path_returns = ray.get(path_return_refs)
    cost = time.time() - start_time
    print(cost)
        
def test_remote_pkl_worker(num_worker=25):
    import time
    from pprint import pprint as pp
    start_time = time.time()
    env = gym.make("Swimmer-v3")
    env_pkl = pickle.dumps(env)
    remote_workers = [RemoteEnv_pkl.remote(env_pkl, idx) for idx in range(num_worker)]
    path_return_refs = [remote_worker.sample.remote() for remote_worker in remote_workers]
    path_returns = ray.get(path_return_refs)
    cost = time.time() - start_time
    print(cost)

def test_serial_env(num_envs=25):

    def sample(env):
        path_return = 0.
        env.reset()
        while True:
            next_obs, rew, done, info = env.step(env.action_space.sample())
            path_return += rew
            if done:
                break
        return path_return
    start_time = time.time()
    env = gym.make("Swimmer-v3")
    env.reset(seed=0)
    env.action_space.seed(0)
    path_returns = []
    for _ in range(num_envs):
        path_returns.append(sample(env))
    cost = time.time() - start_time
    print(cost)
    
test_remote_pkl_worker()
test_remote_worker()
test_serial_env()


# -

def test_gym_seed():
    env_1 = gym.make("Swimmer-v3")
    env_2 = gym.make("Swimmer-v3")
    obs1, obs2 = env_1.reset(seed=10), env_2.reset(seed=10)
    env_1.step(env_1.action_space.sample())
    env_2.step(env_2.action_space.sample())
    obs1, obs2 = env_1.reset(), env_2.reset()
    print(obs1.sum(), obs2.sum())


from contextlib import contextmanager
class test_render():
    def __init__(self):
        self._render = False
    
    @contextmanager
    def render(self):
        self._render = True
        yield 
        self._render = False
test = test_render()
print(test._render)
with test.render():
    print(test._render)
print(test._render)


# +
@ray.remote
def add(cnt_ref):
    return 1 + cnt_ref

x = 1
k = ray.put(x)
ret_ref = add.remote(x_ref)
print(ray.get(ret_ref))
print(k)
# +
config = {'render': False}
config_ref = ray.put(config)

@ray.remote
def remote_config(config_ref):
    return config_ref

remote_config_ref = remote_config.remote(config_ref)
print(ray.get(remote_config_ref))

config['render'] = True
remote_config_ref = remote_config.remote(config_ref)
print(ray.get(remote_config_ref))
# -


