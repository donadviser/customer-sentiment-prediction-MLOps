from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

from src.logger import logging
from src.exceptions import CustomException
import pandas as pd



@pipeline(enable_cache=True)

def training_pipeline(data_path: str)->None:
    """A pipeline to train a model.

    Args:
        data_path (str): The path to the data file.
    """
    try:
        df = ingest_df(data_path)
        X_train, X_test, y_train, y_test = clean_df(df)
        trained_model = train_model(X_train, y_train)
        r2_score, rmse = evaluate_model(trained_model, X_test, y_test)
        logging.info(f"R2 score: {r2_score}")
        logging.info(f"RMSE: {rmse}")
    except Exception as e:
        raise CustomException(e, "Error while training the model")