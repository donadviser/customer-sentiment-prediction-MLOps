import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

from src.logger import logging
from src.exceptions import CustomException
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from zenml import step

@step(enable_cache=False)
def clean_df(df: pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    ]:
    
    """
    Clean, Preprocess and Divide the data

    Returns:
        X_train (np.ndarray)
        X_test (np.ndarray)
        y_train (np.ndarray)
        y_test (np.ndarray)
    """
    try:
        data_cleaning = DataCleaning(df, DataPreProcessStrategy())
        processed_data = data_cleaning.handle_data()

        data_cleaning = DataCleaning(processed_data, DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise CustomException(e, "Errorr while cleaning, preprocessing and dividing data")