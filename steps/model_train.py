from src.logger import logging
from src.exceptions import CustomException
from sklearn.base import RegressorMixin
from src.model_dev import LinearRegressionModel
from src.model_config import ModelNameConfig
from zenml import step

import pandas as pd

@step
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_config: ModelNameConfig,
    )->RegressorMixin:

    """
    Trains the model

    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        X_test (pd.DataFrame): Testing data
        y_test (pd.DataFrame): Testing labels
        model_config (ModelNameConfig): Model configuration

    Returns:
        RegressorMixin: Trained model
    """
    try:
        model = None
        if model_config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            logging.info("Model training completed successfully")
            return trained_model
        else:
            raise ValueError (f"Model {model_config.model_name} not suported")
    except Exception as e:
        CustomException(e, "Error while training the model")
