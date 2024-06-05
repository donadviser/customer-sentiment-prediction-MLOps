import pandas as pd
import joblib
from typing import Tuple, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.feature_engineering import create_preprocessing_pipeline


from src.logger import logging
from src.exceptions import CustomException

def load_preprocessor(file_path: str = 'preprocessor.joblib') -> ColumnTransformer:
    """
    Load the pre-trained preprocessor from a specified file path.

    Args:
        file_path (str): The file path to load the preprocessor from.

    Returns:
        ColumnTransformer: The loaded preprocessor.
    """
    try:
        preprocessor = joblib.load(file_path)
        logging.info(f'Preprocessor loaded from {file_path}')
        return preprocessor
    except Exception as e:
        raise CustomException(e, f"Error in load_preprocessor")

def preprocess_new_data(df: pd.DataFrame, preprocessor: ColumnTransformer) -> Union[pd.DataFrame, None]:
    """
    Preprocess new data using the loaded preprocessor and custom transformations.

    Args:
        df (pd.DataFrame): The new data to preprocess.
        preprocessor (ColumnTransformer): The pre-trained preprocessor.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    try:
        # Obtain the preprocessing pipeline
        pipeline, _, _ = create_preprocessing_pipeline(df)
        
        # Apply custom preprocessing steps
        df_clean = pipeline.transform(df)

        # Apply column-specific transformations
        df_preprocessed = preprocessor.transform(df_clean)

        logging.info("New data preprocessed successfully.")
        return df_preprocessed
    except Exception as e:
        raise CustomException(e, f"Error in preprocess_new_data")

# Example usage during inference
# df_new = pd.read_csv('new_data.csv')
# try:
#     preprocessor = load_preprocessor('preprocessor.joblib')
#     df_new_preprocessed = preprocess_new_data(df_new, preprocessor)
#     # df_new_preprocessed can now be used for inference with the model
# except CustomException as e:
#     logging.error(f"An error occurred: {e}")