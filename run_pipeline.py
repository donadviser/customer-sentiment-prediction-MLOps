from src.logger import logging
from src.exceptions import CustomException
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    try:
        logging.info(f"PIPELINE: Training Pipeline Started")
        #print(Client().active_stack.experiment_tracker.get_tracking_uri())
        data_path = "data/olist_customers_dataset.csv"        
        train_pipeline(data_path)
        logging.info(f"PIPELINE: Training Pipeline completed successfully")
    except Exception as e:
        raise CustomException(e, "Error while training the model")