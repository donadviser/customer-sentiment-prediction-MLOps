from src.logger import logging
from src.exceptions import CustomException
from pipeline.training_pipeline import training_pipeline

if __name__ == "__main__":
    try:
        logging.info(f"Training Pipeline Started")
        data_path = "data/olist_customers_dataset.csv"
        training_pipeline(data_path)
        logging.info(f"Training Pipeline successfully")
    except Exception as e:
        raise CustomException(e, "Error while training the model")
