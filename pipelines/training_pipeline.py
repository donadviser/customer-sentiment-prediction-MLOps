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
def train_pipeline(data_path: str)->None:
    """A pipeline to train a model.

    Args:
        data_path (str): The path to the data file.
    """
    try:
        logging.info("Starting training pipeline")
    
        df = ingest_df(data_path)
        print(df)
        print(df.columns)
        target_col = "satisfaction"
        X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor = clean_df(df, target_col)
        classifier_model = train_model(X_train_preprocessed, y_train, X_test_preprocessed, y_test)

        
        """
        accuracy, precision_score, recall_score, f1_score, confusion_matrix, classification_report  = evaluate_model(classifier_model, X_test_preprocessed, y_test)
        logging.info(f"accuracy: {accuracy}")
        logging.info(f"precision_score: {precision_score}")
        logging.info(f"recall_score: {recall_score}")
        logging.info(f"f1_score: {f1_score}")
        logging.info(f"confusion_matrix: {confusion_matrix}")
        logging.info(f"classification_report: {classification_report}")
        logging.info("Training pipeline completed successfully")

        joblib.dump(preprocessor, 'artefacts/preprocessor.joblib')
        joblib.dump(X_train_preprocessed, 'artefacts/X_train.joblib')
        joblib.dump(X_test_preprocessed, 'artefacts/X_test.joblib')
        joblib.dump(y_train, 'artefacts/y_train.joblib')
        joblib.dump(y_test, 'artefacts/y_test.joblib')"""

    except Exception as e:
        raise CustomException(e, "Error while training the model")