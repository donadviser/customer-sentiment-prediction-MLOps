from src.logger import logging
from src.exceptions import CustomException
from pipeline.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    try:
        logging.info(f"Training Pipeline Started")
        data_path = "data/olist_customers_dataset.csv"
        print(Client().active_stack.experiment_tracker.get_tracking_uri())
        training_pipeline(data_path)
        logging.info(f"Training Pipeline successfully")
    except Exception as e:
        raise CustomException(e, "Error while training the model")
