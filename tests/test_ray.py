# -*- coding: utf-8 -*-
import ray
import numpy as np
import time
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




