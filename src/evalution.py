from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
import numpy as np

class Evaluation(ABC):
    """
    Abstract base defingin strategy for handling data
    """
    @abstractmethod
    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    """
    Strategy for evaluating mean squared error
    """

    def calcualate_scores(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            logging.info("y_pred: {y_pred}")
            mse = mean_squared_error(y_test, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            raise CustomException(e, "Error while calculating MSE")


class R2(Evaluation):
    """
    Strategy for evaluating R2 score
    """

    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_test, y_pred)
            logging.info(f"R2 Scores: {r2}")
            return r2
        except Exception as e:
            raise CustomException(e, "Error while calculating R2 Score")

class RMSE(Evaluation):
    """
    Strategy for evaluating root mean squared error
    """

    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_test, y_pred)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
           raise CustomException(e, "Error while calculating RMSE")