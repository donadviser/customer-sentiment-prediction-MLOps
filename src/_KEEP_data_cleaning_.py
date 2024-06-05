import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
from typing import Tuple, Union
from typing_extensions import Annotated


class DataStrategy(ABC):
    """
    Abstract base defining strategy for handling data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        pass


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
                          'product_photos_qty', 'review_score', 'product_length_cm', 'product_height_cm', 
                          'product_width_cm', 'price', 'freight_value']

        try:
            id_cols = data.filter(like='_id').columns.tolist()
            data = (data
                    .dropna(thresh=len(data) * 0.5, axis=1)
                    .assign(**{col: pd.to_datetime(data[col], errors='coerce') for col in datetime_cols})
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
                    .drop(columns=id_cols)                    
                    .drop(columns = datetime_cols)
                   )
            logging.info(data.head())
            print(f"data.columns: {data.columns}")
            print(data.head())
            print(f"DataPreProcessStrategy completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, "Error while processing data")


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
            print(f"Inside DataDivideStrategy")
            print(data.columns)
            X = data.drop(columns=["satisfaction"])
            y = data["satisfaction"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"DataDivideStrategy completed successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, "Error while dividing data")


class DataEncodeStrategy(DataStrategy):
    """
    Strategy for encoding data.
    """
    def handle_data(self, data: Tuple[pd.DataFrame, pd.DataFrame], target_col: str) -> Tuple[
        Annotated[pd.DataFrame, "X_train_processeding"],
        Annotated[pd.DataFrame, "X_test_processeding"],
        Annotated[str, "preprocessor"]
    ]:
        """
        Encode data.

        Args:
            data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of training and testing data

        Returns:
            Tuple[
                Annotated[pd.DataFrame, "X_train_processed"],
                Annotated[pd.DataFrame, "X_test_processed"],
                Annotated[ColumnTransformer, "preprocessor"]
            ]: Encoded training and testing data, and the preprocessor.
        """
        try:
            X_train, X_test = data
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            numerical_cols = X_train.select_dtypes(include=[np.number]).columns.difference([target_col]).tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols),
                ])

            X_train_preprocessed = preprocessor.fit_transform(X_train)
            X_test_preprocessed = preprocessor.transform(X_test)

            logging.info("Data encoded successfully.")
            return X_train_preprocessed, X_test_preprocessed, preprocessor
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


def preprocess_and_split_data(df: pd.DataFrame, target_col: str = 'satisfaction') -> Tuple:
    """
    Preprocess the data and split into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        Tuple: Preprocessed training and testing data, and the preprocessor.
    """
    try:
        data_preprocessor = DataCleaning(df, DataPreProcessStrategy())
        preprocessed_data = data_preprocessor.handle_data()
        
        data_divider = DataCleaning(preprocessed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_divider.handle_data()

        data_encoder = DataCleaning((X_train, X_test), DataEncodeStrategy(), target_col)
        X_train_preprocessed, X_test_preprocessed, preprocessor = data_encoder.handle_data()
    
        logging.info("Data preprocessed, split, and encoded successfully.")
        return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor
    except Exception as e:
        raise CustomException(e, "Error in preprocess_and_split_data")


'''if __name__ == "__main__":
    df = pd.read_csv('/Users/don/github-projects/pre-purchase-sentiment-prediction-MLOps/data/olist_customers_dataset.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split_data(df)

    joblib.dump(preprocessor, 'artefacts/preprocessor.joblib')
    joblib.dump(X_train, 'artefacts/X_train.joblib')
    joblib.dump(X_test, 'artefacts/X_test.joblib')
    joblib.dump(y_train, 'artefacts/y_train.joblib')
    joblib.dump(y_test, 'artefacts/y_test.joblib')

    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"preprocessor: {preprocessor}")'''
