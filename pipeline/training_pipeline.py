from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
import joblib



#@pipeline(enable_cache=False)

def training_pipeline(data_path: str)->None:
    """A pipeline to train a model.

    Args:
        data_path (str): The path to the data file.
    """
    try:
        df = ingest_df(data_path)
        print(f"data_path: {data_path}")
        print(f"type(df): {type(df)}")
        train_array, test_array, preprocessor = clean_df(df)
        #print(f"Inside training pipeline type(X_train): {type(X_train)}")
        '''joblib.dump(preprocessor, 'artefacts/preprocessor.joblib')
        joblib.dump(X_train, 'artefacts/X_train.joblib')
        joblib.dump(X_test, 'artefacts/X_test.joblib')
        joblib.dump(y_train, 'artefacts/y_train.joblib')
        joblib.dump(y_test, 'artefacts/y_test.joblib')'''

        X_train = train_array[:, : -1]
        y_train = train_array[:, -1]
        X_test = test_array[:, :-1]
        y_test = test_array[:, -1]

        trained_model = train_model(X_train, y_train)
        """r2_score, rmse, mse = evaluate_model(trained_model, X_test, y_test)
        logging.info(f"R2 score: {r2_score}")
        logging.info(f"RMSE: {rmse}")
        logging.info(f"MSE: {mse}")"""
    except Exception as e:
        raise CustomException(e, "Error while training the model")