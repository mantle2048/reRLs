from .base_sampler import BaseSampler
from .simple_sampler import SimpleSampler, rollout, rollouts
from .remote_sampler import RemoteSampler
from .vector_sampler import VectorSampler

def get(num_workers=1, num_envs=1):
    """Returns a sampler.
    Arguments:
        num_worker.
    Returns:
        A remote sampler if num_worker > 1 else a simple worker
    """
    if num_workers > 1 and num_envs == 1:
        return RemoteSampler
    elif num_workers == 1 and num_envs == 1:
        return SimpleSampler
    elif num_workers == 1 and num_envs > 1:
        return VectorSampler
    elif num_workers > 1 and num_envs > 1:
        raise TypeError(
            f"Remote Sampler with VectorEnv is not supported not")
    else:
        raise TypeError(
            f"Not Supported Sampler style")

