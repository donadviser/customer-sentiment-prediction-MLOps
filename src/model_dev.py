from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin


class Model(ABC):
    """
    Abstract base defingin strategy for handling data
    """
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """
        pass

class LinearRegressionModel(Model):
    """
    Strategy for training linear regression model
    """

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RegressorMixin:
        """
        Train the model

        Args:
            X_train (np.ndarray): Training data
            y_train (np.ndarray): Training labels
        """
        try:
            model = LinearRegression(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed successfully")
            return model
        except Exception as e:
            raise CustomException(e, "Error while training the model")

