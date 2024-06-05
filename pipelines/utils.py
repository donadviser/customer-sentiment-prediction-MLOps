import logging
import os
import sys

# Get the path of the parent folder
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent folder to the Python path
sys.path.append(parent_folder_path)

import pandas as pd
import joblib
from src.logger import logging
from src.exceptions import CustomException

from src.data_cleaning import (
    DataCleaning, 
    DataDateTimeConverter,
    DataPreProcessStrategy,
)

def get_data_for_test(target_col: str='satisfaction'):
    try:
        logging.info("Starting to get encoded data for test")
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)

        data_preprocessed = DataCleaning(df, DataDateTimeConverter())
        data_preprocessed = data_preprocessed.handle_data()
        #print(f"DataDateTimeConverter:\n{data_preprocessed.head()}")

        data_preprocessed = DataCleaning(data_preprocessed, DataPreProcessStrategy())
        data_preprocessed = data_preprocessed.handle_data()
        #print(f"DataPreProcessStrategy:\n{data_preprocessed.columns}")

        X_test_preprocessed = data_preprocessed.drop(columns=target_col)

        # load the pipeline from the file
        loaded_preprocess_pipeline = joblib.load('artefacts/preprocessor.joblib')

        # Use the loaded preprocessing pipeline to transform the new data
        #print(loaded_preprocess_pipeline)
        X_test_encoded = loaded_preprocess_pipeline.transform(X_test_preprocessed)
        #print(X_test_encoded.shape)
    
        logging.info("Conpleted preprocessing pipeline for the test dataset")
        
        return X_test_encoded
    except Exception as e:
        raise CustomException(e, "Failed to load preprocessing pipeline to clean the inference dataset")
    

"""if __name__ == "__main__":
    result = get_data_for_test()
    print(result[:5,:5])"""