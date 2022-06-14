import os
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple
from math import sqrt

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)



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


def preprocess():
    journeys = pd.read_csv('../data/journeys.csv')
    journeys['Trip Start At Local Time'] = pd.to_datetime(journeys['Trip Start At Local Time'])
    journeys['Trip End At Local Time'] = pd.to_datetime(journeys['Trip End At Local Time'])
    journeys['duration'] = journeys['Trip End At Local Time'] - journeys['Trip Start At Local Time']
    journeys['duration'] = journeys['duration'].astype('timedelta64[h]')
    journeys['datetime'] = journeys['Trip Start At Local Time'].copy()
    journeys['city'] = journeys['Car Parking Address City'].copy()
    journeys['trip_price'] = (
        journeys['Trip Sum Trip Price']
        .str.replace('$', '', regex=False)
        .str.replace(',', '', regex=False)
        .astype(float)
    )
    return journeys


def split_data(data, **split_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_cols = ['city', 'datetime', 'duration']
    y_col = 'trip_price'
    X = data[X_cols].copy()
    y = data[y_col].copy()
    
    return train_test_split(X, y, **split_kwargs)


def full_pipeline():
    dates_pipe = Pipeline(
        [
            ('date_selector', DataFrameSelect('datetime')),
            ('handler', DateTimeHandler()),
        ]
    )
    city_pipe = Pipeline(
        [
            ('city_selector', DataFrameSelect(['city']))
        ]
    )
    union = FeatureUnion(
        [
            ('dates', dates_pipe),
            ('cities', city_pipe)
        ]
    )
    categoricals = Pipeline(
        [
            ('union', union),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]
    )
    full_pipe = FeatureUnion(
        [
            ('cats', categoricals),
            ('duration', DataFrameSelect(['duration']))
        ]
    )
    return Pipeline(
        [
            ('pipe', full_pipe),
            ('cat', CatBoostRegressor(verbose=False))
        ]
    )


def regression_metrics(y_true, y_pred):
    return pd.Series(
        {
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    )


def metrics_df(y_train, y_pred_train, y_test, y_pred_test):
    train_metrics = regression_metrics(y_train, y_pred_train)
    test_metrics = regression_metrics(y_test, y_pred_test)
    df = pd.concat([train_metrics, test_metrics], axis=1)
    df.columns = ['train', 'test']
    return df


def plot_resid(y_true, y_pred, title):
    resid = y_true - y_pred
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        y=resid,
        x=y_pred
    );
    plt.axhline(resid.mean(), color='r');
    plt.xlabel('$\hat{y}$');
    plt.ylabel('u');
    plt.title(title);
    plt.legend(['residuals', 'mean'])
    plt.savefig(f'../models/{title}.png')


def make_markdown_report(metrics):
    return f"""
    | Metric | Train | Test |
    | ------ | ----- | ---- |
    | MAE    | {metrics.loc['MAE', 'train']:.2f} | {metrics.loc['MAE', 'test']:.2f} |
    | MSE    | {metrics.loc['MSE', 'train']:.2f} | {metrics.loc['MSE', 'test']:.2f} |
    | RMSE   | {metrics.loc['RMSE', 'train']:.2f} | {metrics.loc['RMSE', 'test']:.2f} |
    | MAPE   | {metrics.loc['MAPE', 'train']:.2f} | {metrics.loc['MAPE', 'test']:.2f} |
    | R2     | {metrics.loc['R2', 'train']:.2f} | {metrics.loc['R2', 'test']:.2f} |
    """


if __name__ == '__main__':
    if not os.path.exists('../models/price_model.pkl'):
        os.makedirs('../models', exist_ok=True)
        
        print("Preprocessing data...")
        data = preprocess()
        X_train, X_test, y_train, y_test = split_data(data, test_size=0.2, random_state=42)
        model = full_pipeline()
        print("Fitting model...")
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        metrics = metrics_df(y_train, y_pred_train, y_test, y_pred_test)
        with open('../models/metrics.md', 'w') as f:
            f.write(make_markdown_report(metrics))
        print(metrics)
        metrics.to_csv('../models/metrics.csv', index=False)
        
        plot_resid(y_test, y_pred_test, 'Test')
        plot_resid(y_train, y_pred_train, 'Train')
        write_path = '../models/price_model.pkl'
        with open(write_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Saving model to {write_path}")