from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from zenml import step


class IngestData:
    """
    Ingests data from a source.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        """
        Gets data from a source.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

#@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a source.

    Args:
        data_path (str): The path to the data source.

    Returns:
        pd.DataFrame: The ingested data.
    """

    try:
        logging.info(f"Starting data ingestion")
        data_ingestion = IngestData(data_path)
        df_raw = data_ingestion.get_data()
        logging.info(f"Data ingestion completed successfully")
        return df_raw
    
    except Exception as e:
        raise CustomException(e, "Error while ingesting data")
    
   