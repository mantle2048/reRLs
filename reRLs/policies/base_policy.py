import abc
import numpy as np


class BasePolicy(abc.ABC):

    @abc.abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, obs: np.ndarray, acs: np.ndarray, **kwargs) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError
