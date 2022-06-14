import pickle

from typing import Union
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
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


class DataFrameSelect(BaseEstimator, TransformerMixin):
    
    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.column_names]
    
class DateTimeHandler(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        date_serie = pd.to_datetime(x)
        features = []
        features.append(date_serie.dt.month)
        features.append(date_serie.dt.day)
        features.append(date_serie.dt.weekday)
        features.append(date_serie.dt.hour)
        features.append(date_serie.dt.minute)
        df = pd.concat(features, axis=1)
        df.columns = ['month', 'day', 'weekday', 'hour', 'minute']

        return df