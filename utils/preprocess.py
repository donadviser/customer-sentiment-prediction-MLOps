from typing import Union
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import logging
from src.exceptions import CustomException


# Custom transformers
class DropMissingThreshold(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns with missing values above a certain threshold.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropMissingThreshold':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.dropna(thresh=len(X) * self.threshold, axis=1)
        except Exception as e:
            raise CustomException(e, f"Error in DropMissingThreshold")
        
    
class DateTimeConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to convert specified columns to datetime.
    """
    def __init__(self, datetime_cols: list):
        self.datetime_cols = datetime_cols
        
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DateTimeConverter':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X=(X.assign(**{col: pd.to_datetime(X[col], errors='coerce') for col in self.datetime_cols}))
            return X
        except Exception as e:
            raise CustomException(e, f"Error in DateTimeConverter")
        

    
class DropRedundantColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to drop redundant columns from the dataset.
    """
    def __init__(self, redundant_cols: list):
        self.redundant_cols = redundant_cols
    
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropRedundantColumns':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.drop(columns=self.redundant_cols)
        except Exception as e:
            raise CustomException(e, f"Error in DropRedundantColumns")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer to engineer new features from the dataset.
    """
    def fit(self, X: pd.DataFrame, y: None = None) -> 'FeatureEngineer':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X['time_to_delivery'] = (X['order_delivered_customer_date'] - X['order_approved_at']).dt.days
            X['order_processing_time'] = (X['order_approved_at'] - X['order_purchase_timestamp']).dt.days
            X['estimated_vs_actual_shipping'] = (X['order_estimated_delivery_date'] - X['order_delivered_customer_date']).dt.days
            X['product_volume_m3'] = (X['product_length_cm'] * X['product_width_cm'] * X['product_height_cm']) / 1000000
            X['satisfaction'] = (X['review_score'] >= 4).astype(int)
            X['order_value'] = X['price'] + X['freight_value']
            X['late_delivery'] = (X['order_delivered_customer_date'] > X['order_estimated_delivery_date']).astype(int)
            
            X['order_month'] = X['order_purchase_timestamp'].dt.month
            X['order_day'] = X['order_purchase_timestamp'].dt.dayofweek
            X['order_hour'] = X['order_purchase_timestamp'].dt.hour
            return X
        except Exception as e:
            raise CustomException(e, f"Error in FeatureEngineer")
        
    

class NADropper:
    """Support class to drop NA values in sklearn Pipeline."""

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.dropna()


class ColumnsDropper:
    """Support class to drop specific columns in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]):
        return X.drop(columns=self.columns)


class DataFrameCaster:
    """Support class to cast type back to pd.DataFrame in sklearn Pipeline."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)
