import pickle

from typing import Union

import pandas as pd

from andela.blueprints import Payload, Model


class PricePayload(Payload):
    city_name: str
    journey_starting_datetime: str
    journey_duration_hours: Union[int, float]

    def __post_init__(self):
        self.df = self.transform_raw_data()

    def transform_raw_data(self):
        return pd.DataFrame([self.__dict__])


class PriceModel(Model):
    def __init__(self, path: str) -> None:
        self.path = path

    def load_model(self):
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data):
        return self.model.predict(data)

    def transform(self, data):
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['journey_starting_datetime'])
        df.rename(
            columns={
                'duration_hours': 'duration',
                'city_name': 'city'
            },
            inplace=True
        )
        return df[['datetime', 'duration', 'city']].copy()
