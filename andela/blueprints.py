from dataclasses import dataclass
from abc import abstractmethod, ABC


@dataclass
class Payload(ABC):
    """Abstract Data Class Payload

    Blueprint for handling contents from a request.

    Specify all data fields required, with their corresponding data types.

    Abstract Methods
    ----------------
    __post_init__

    transform_raw_data
    """

    trip_id: str

    @abstractmethod
    def __post_init__(self):
        pass

    @abstractmethod
    def transform_raw_data(self):
        raise NotImplementedError


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

