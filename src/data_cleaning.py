import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, List
from typing_extensions import Annotated
from utils.preprocess import DropMissingThreshold, DateTimeConverter, DropRedundantColumns, FeatureEngineer, DataFrameCaster


def create_preprocessing_pipeline(
        df: pd.DataFrame,
        drop_missing_threshold: Optional[bool] = None,
        drop_na: Optional[bool] = None,
        drop_columns: Optional[List[str]] = None,
        convert_datetime: Optional[bool] = None,
        feature_engineering: Optional[bool] = None,
        ):
    """
    Create a preprocessing pipeline and define the columns for numeric and categorical transformers.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Pipeline, list, list]: A tuple containing the preprocessing pipeline,
                                     list of numerical columns, and list of categorical columns.
    """
    try:
        redundant_cols = [
            'product_name_lenght', 'product_description_lenght', 
            'product_photos_qty', 'review_score', 'order_delivered_customer_date',
            'order_approved_at', 'order_estimated_delivery_date', 'order_purchase_timestamp',
            'product_length_cm', 'product_height_cm', 'product_width_cm',
            'price', 'freight_value'
        ]

        numerical_cols = ['payment_value', 'order_value', 'product_weight_g', 'product_volume_m3', 'order_item_id', 'estimated_vs_actual_shipping',
                  'time_to_delivery', 'order_processing_time', 'payment_installments','customer_zip_code_prefix']
        categorical_cols = ['order_month', 'order_day', 'order_hour','late_delivery']
        datetime_cols = ['order_delivered_customer_date', 'order_approved_at', 'order_estimated_delivery_date', 'order_purchase_timestamp']

        # We use the sklearn pipeline to chain together multiple preprocessing steps
        preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
        if drop_na:
            preprocess_pipeline.steps.append(('drop_na', FunctionTransformer(lambda x: x.dropna(), validate=False)))

        if convert_datetime:
            preprocess_pipeline.steps.append(('convert_datetime', DateTimeConverter(datetime_cols=datetime_cols)))

        if drop_missing_threshold:
            preprocess_pipeline.steps.append(('drop_missing_threshold', DropMissingThreshold(threshold=0.5)))

        if feature_engineering:
            preprocess_pipeline.steps.append(('feature_engineering', FeatureEngineer()))

        if drop_columns:
            preprocess_pipeline.steps.append(('drop_columns', DropRedundantColumns(redundant_cols=redundant_cols)))

        return preprocess_pipeline 
    except Exception as e:
        raise CustomException(e, f"Error in create_preprocessing_pipeline")

def preprocess_and_split_data(df: pd.DataFrame, target_col: str = 'satisfaction') ->  Tuple[
    Annotated[pd.DataFrame, "X_train_preprocessed"],
    Annotated[pd.DataFrame, "X_test_preprocessed"],
    Annotated[Pipeline, "preprocess_pipeline"],
]:
    """
    Preprocess the data and split into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        Tuple: Preprocessed training and testing data, and the preprocessor.
    """
    try:
        preprocess_pipeline = create_preprocessing_pipeline(df, drop_missing_threshold = True,
                                                            drop_na = True, drop_columns=True,
                                                            convert_datetime=True, feature_engineering=True)
        
        data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
        data_train = pd.DataFrame(data_train, columns=df.columns)
        data_test = pd.DataFrame(data_test, columns=df.columns)

        preprocess_pipeline.steps.append(("cast", DataFrameCaster(data_train.columns)))
        input_feature_train_array = preprocess_pipeline.fit_transform(data_train)
        input_feature_test_array = preprocess_pipeline.transform(data_test)

        print(input_feature_train_array.columns)
        input_feature_train_df = data_train.drop(target_col, axis=1)
        target_feature_train_df = data_test[target_col]

        input_feature_test_df = data_train.drop(target_col, axis=1)
        target_feature_test_df = data_test[target_col]

        logging.info("Applying preprocessing object on training and test dataframes.")


        

        print(input_feature_train_array.columns)

        logging.info("concatenating preprocessed features and target variabled for boeth traing and test arrays")
        train_array = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
        test_array = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

       

        logging.info("Data preprocessed and split into training and testing sets successfully.")
        return (train_array, test_array, preprocess_pipeline)
    except Exception as e:
        raise CustomException(e, f"Error in preprocess_and_split_data")
    

if __name__ == "__main__":
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
    print(f"preprocessor: {preprocessor}")
