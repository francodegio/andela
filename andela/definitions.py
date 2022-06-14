import pickle

from typing import Union
from pydantic import BaseModel

import pandas as pd

from andela.blueprints import Model


class TripPayload(BaseModel):
    city_name: str
    journey_starting_datetime: str
    journey_duration_hours: Union[int, float]


class PriceModel(Model):
    def __init__(self, path: str) -> None:
        self.path = path
        self.load_model()

    def load_model(self):
        with open(self.path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data):
        return self.model.predict(data)

    def transform(self, data):
        df = data.copy()
        df['datetime'] = pd.to_datetime(df['journey_starting_datetime'])
        df = df.rename(
            columns={
                'journey_duration_hours': 'duration',
                'city_name': 'city'
            }
        )
        return df[['datetime', 'duration', 'city']].copy()
