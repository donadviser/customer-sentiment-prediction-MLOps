from src.logger import logging
from src.exceptions import CustomException
from sklearn.compose import ColumnTransformer
from src.data_cleaning import (
    DataCleaning,
    DataDateTimeConverter,
    DataPreProcessStrategy,
    DropMissingThreshold,
    DataDivideStrategy,
    DataEncodeStrategy
    )


import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from zenml import step


@step
def clean_df(df: pd.DataFrame, target_col: str) -> Tuple[
    Annotated[np.ndarray, "X_train_encoded"],
    Annotated[np.ndarray, "X_test_encoded"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
    Annotated[ColumnTransformer, "preprocess_pipeline"]
]:
    
    try:
        data_preprocessed = DataCleaning(df, DataDateTimeConverter())
        data_preprocessed = data_preprocessed.handle_data()
        #print(f"\nDataDateTimeConverter:\n{data_preprocessed.columns}")

        data_preprocessed = DataCleaning(data_preprocessed, DataPreProcessStrategy())
        data_preprocessed = data_preprocessed.handle_data()
        #print(f"/nDataPreProcessStrategy:\n{data_preprocessed.shape}")

        data_preprocessed = DataCleaning(data_preprocessed, DropMissingThreshold())
        data_preprocessed = data_preprocessed.handle_data()
        #print(f"\nDropMissingThreshold\n{data_preprocessed.shape}")
        #print(data_preprocessed.columns)
        #print(data_preprocessed.head())

        data_divider = DataCleaning(data_preprocessed, DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_divider.handle_data()
        #print(f"\nDataDivideStrategy:\n{X_train.shape}")
        #print(X_test.shape)
        #print(y_train.shape)
        #print(y_test.shape)
        #print(y_train.head())

        data_encoder = DataCleaning((X_train), DataEncodeStrategy(), target_col)
        preprocess_pipeline = data_encoder.handle_data()

        X_train_encoded = preprocess_pipeline.fit_transform(X_train, y_train)
        X_test_encoded = preprocess_pipeline.transform(X_test)
        

        # Convert y_train and y_test to np.ndarray
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        return X_train_encoded, X_test_encoded, y_train, y_test, preprocess_pipeline
    except Exception as e:
        raise CustomException(e, f"Error in clean_df")