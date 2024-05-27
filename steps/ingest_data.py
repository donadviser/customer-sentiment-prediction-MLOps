from src.logger import logging
from src.exceptions import CustomException
from zenml import step

import pandas as pd

class IngestData:

    def __init__(self, data_path: str)-> None:
        self.data_path = data_path
    
    def get_data(self) -> pd.DataFrame:
        """Loads the data from the given path.

        Args:
            data_path (str): The path to the data file.
        """
        logging.info(f"Loading data from {self.data_path}")        
        return pd.read_csv(self.data_path)
    
        
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """Loads the data from the given path.

    Args:
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        logging.info(f"Loaded data successfully")
        return df
    except Exception as e:
        raise CustomException(e, "Error while ingesting the data")