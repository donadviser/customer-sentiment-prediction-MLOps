import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import joblib
from src.logger import logging
from src.exceptions import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_cleaning import preprocess_and_split_data
from zenml import step

@step(enable_cache=False)
def clean_df(df: pd.DataFrame) ->Tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    
    """
    Clean, Preprocess, Divide and Encode the data

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns: 
        Tuple: Preprocessed training and testing data, and the preprocessor:
        X_train (np.ndarray)
        X_test (np.ndarray)
        y_train (np.ndarray)
        y_test (np.ndarray)
        preprocessor (str)
    """

    try:
        logging.info("Data Clean, Preprocess, Divide and Encode starting")
        target_col: str = 'satisfaction'

        X_train_preprocessed, X_test_preprocessed, preprocess_pipeline = preprocess_and_split_data(df)
    
        logging.info("Data cleaned, preprocessed, split, and encoded successfully.")

        return X_train_preprocessed, X_test_preprocessed, preprocess_pipeline
    except Exception as e:
        raise CustomException(e, "Errorr while cleaning, preprocessing, dividing, and encoding the data")
    

if __name__ == "__main__":
    df = pd.read_csv('/Users/don/github-projects/pre-purchase-sentiment-prediction-MLOps/data/olist_customers_dataset.csv')
    X_train, X_test, y_train, y_test, preprocessor = clean_df(df)

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