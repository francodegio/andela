from dataclasses import dataclass
from abc import abstractmethod, ABC


class Model(ABC):
    """Model Abstract Class

    Blueprint for representing ML models that need to be loaded into memory.

    Parameters
    ----------
    path:
        The model path to load.

    Abstract Methods
    ----------------
    __init__

    find_latest_model

    transform

    predict
    """

    @abstractmethod
    def __init__(self, path: str) -> None:
        self.path = path

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, data):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data):
        raise NotImplementedError

