import abc
from collections import OrderedDict
from contextlib import contextmanager

class BaseSampler(object):
    def __init__(self, env, trainer_config):
        self._env = env

        self._seed = trainer_config['seed']
        self._max_path_length = trainer_config['ep_len']
        self._deterministic = trainer_config['deterministic']
        self._render = False
        self._render_mode = ('rgb_array',)

    @contextmanager
    def render(self):
        self._render = True
        yield
        self._render = False

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError

    def close(self):
        self._env.close()

    def get_diagnostics(self):
        diagnostics = OrderedDict({})
        return diagnostics

    @property
    def env(self):
        return self._env
