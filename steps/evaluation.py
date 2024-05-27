from src.logger import logging
from src.exceptions import CustomException
from zenml import step
from src.evalution import R2, MSE, RMSE
from sklearn.base import RegressorMixin

import pandas as pd
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   )->Tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"]
                   ]:
    
    """
    Evaluates the model

    Args:
        model (RegressorMixin): Trained model
    
    Returns:
        None
    """
    try:
        y_pred = model.predict(X_test)

        r2_ = R2()
        r2_score = r2_.calculate_scores(y_test, y_pred)

        #mse_ = MSE()
       #mse = mse_.calculate_scores(y_test, y_pred)

        rmse_ = RMSE()
        rmse = rmse_.calculate_scores(y_test, y_pred)

        return r2_score, rmse
    except Exception as e:
        raise CustomException(e, "Error while evaluating the model")