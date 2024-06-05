import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
from typing import Tuple, Union
from typing_extensions import Annotated
from feature_engine.encoding import MeanEncoder


class DataStrategy(ABC):
    """
    Abstract base defining strategy for handling data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        pass


class DataDateTimeConverter(DataStrategy):
    """
    Strategy for converting datetime columns.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        """
        Converts datetime columns to datetime type.

        Args:
            data (pd.DataFrame): Data to be converted.

        Returns:
            Union[pd.DataFrame, pd.Series]: Converted data.
        """
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 
                         'order_delivered_carrier_date', 'order_delivered_customer_date', 
                         'order_estimated_delivery_date', 'shipping_limit_date']
    
        
        try:
            data = data.assign(**{col: pd.to_datetime(data[col], errors='coerce') for col in datetime_cols})
            logging.info(f"Converting datetime columns completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, "Error converting datetime columns")



class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        """
        Preprocess data.

        Args:
            data (pd.DataFrame): Data to be preprocessed.

        Returns:
            Union[pd.DataFrame, pd.Series]: Preprocessed data.
        """
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 
                         'order_delivered_carrier_date', 'order_delivered_customer_date', 
                         'order_estimated_delivery_date', 'shipping_limit_date']
        
        redundant_cols = ['product_name_lenght', 'product_description_lenght', 
                          'product_photos_qty','review_score', 'product_length_cm', 'product_width_cm', 'product_height_cm', 
                          'product_weight_g', 'order_item_id', 'estimated_vs_actual_shipping',
                          'time_to_delivery', 'order_processing_time','product_id', 'order_id', 'customer_id',
                          'customer_unique_id','customer_zip_code_prefix','seller_id','review_comment_message','customer_city','price','freight_value'
                          ]
        
        try:
            data = (data
                    #.dropna(thresh=len(data) * 0.5, axis=1)
                    #.assign(**{col: pd.to_datetime(data[col], errors='coerce') for col in datetime_cols})
                    .assign(
                        time_to_delivery=lambda x: (x['order_delivered_customer_date'] - x['order_approved_at']).dt.days,
                        order_processing_time=lambda x: (x['order_approved_at'] - x['order_purchase_timestamp']).dt.days,
                        estimated_vs_actual_shipping=lambda x: (x['order_estimated_delivery_date'] - x['order_delivered_customer_date']).dt.days,
                        product_volume_m3=lambda x: (x['product_length_cm'] * x['product_width_cm'] * x['product_height_cm']) / 1000000,
                        satisfaction=lambda x: (x['review_score'] >= 4).astype(int),
                        order_value=lambda x: x['price'] + x['freight_value'],
                        late_delivery=lambda x: (x['order_delivered_customer_date'] > x['order_estimated_delivery_date']).astype(int),
                        order_month=lambda x: x['order_purchase_timestamp'].dt.month,
                        order_day=lambda x: x['order_purchase_timestamp'].dt.dayofweek,
                        order_hour=lambda x: x['order_purchase_timestamp'].dt.hour
                        )
                    .dropna()
                    .drop(columns=redundant_cols)
                    .drop(columns=datetime_cols)


                   )
            print(f"DataPreProcessStrategy completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, "Error while processing data")


class DropMissingThreshold(DataStrategy):
    """
    Strategy for dropping columns with missing threshold.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        threshold = 0.5
        try:
            data = data.dropna(thresh=len(data) * threshold, axis=1)
            logging.info(f"Dropping Missing Threshod of {threshold} completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, f"Error in DropMissingThreshold")
        

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        """
        Divide data.

        Args:
            data (pd.DataFrame): Data to be divided.

        Returns:
            Union[pd.DataFrame, pd.Series]: Divided data.
        """
        try:
            print(f"DataDivideStrategy started")
            X = data.drop(columns="satisfaction")
            y = data["satisfaction"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"DataDivideStrategy completed successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, "Error while dividing data")


class DataEncodeStrategy(DataStrategy):
    """
    Strategy for encoding data.
    """
    def handle_data(self, data: Tuple[pd.DataFrame, pd.DataFrame], target_col: str) -> Annotated[str, "preprocessor"]:
        """
        Encode data.

        Args:
            data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of training data

        Returns:
                Annotated[ColumnTransformer, "preprocessor"]: Encoded training and testing data, and the preprocessor.
        """
        try:
            logging.info("Data encoded started.")
            X_train = data
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('mean_encoder', MeanEncoder())
            ])

            numerical_cols = X_train.select_dtypes(include=[np.number]).columns.difference([target_col]).tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            print(f"numerical_cols: {numerical_cols}")
            print(f"categorical_cols: {categorical_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols),
                ])
            

            logging.info("Data encoded successfully.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, "Error in DataEncodeStrategy")
        

class DataCleaning:
    """
    Class for cleaning data: Clean, Preprocess and Divide the data.
    """
    def __init__(self, data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]], strategy: DataStrategy, target_col: str = None):
        self.data = data
        self.strategy = strategy
        self.target_col = target_col

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple]:
        """
        Clean, Preprocess and Divide the data.

        Returns:
            Union[pd.DataFrame, pd.Series, Tuple]: Clean, Preprocessed and Divided data.
        """
        try:
            if self.target_col:
                return self.strategy.handle_data(self.data, self.target_col)
            return self.strategy.handle_data(self.data)
        except Exception as e:
            raise CustomException(e, "Error while cleaning, preprocessing and dividing data")