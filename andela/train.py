import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin



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