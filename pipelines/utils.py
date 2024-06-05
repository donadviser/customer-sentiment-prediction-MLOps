import logging

import pandas as pd
from src.logger import logging
#from src.exception import CustomException

from src.data_cleaning import (
    DataCleaning,
    DataDateTimeConverter,
    DataPreProcessStrategy,
    DropMissingThreshold,
    DataEncodeStrategy
    )



def get_data_for_test(target_col: str='satisfaction'):
    try:
        df = pd.read_csv("./data/olist_customers_dataset.csv")
        df = df.sample(n=100)

        data_preprocessed = DataCleaning(df, DataDateTimeConverter())
        data_preprocessed = data_preprocessed.handle_data()
        print(f"DataDateTimeConverter:\n{data_preprocessed.columns}")

        data_preprocessed = DataCleaning(data_preprocessed, DataPreProcessStrategy())
        data_preprocessed = data_preprocessed.handle_data()
        print(f"DataPreProcessStrategy:\n{data_preprocessed.shape}")

        data_preprocessed = DataCleaning(data_preprocessed, DropMissingThreshold())
        data_preprocessed = data_preprocessed.handle_data()
        print(f"DropMissingThreshold:\n{data_preprocessed.shape}")
        print(data_preprocessed.columns)
        print(data_preprocessed.head())

        X_test_preprocessed = data_preprocessed.drop(columns=target_col)
        y_test = data_preprocessed[target_col]

        data_encoder = DataCleaning((data_preprocessed), DataEncodeStrategy(), target_col)
        preprocessor = data_encoder.handle_data()

        X_train_preprocessed = preprocessor.fit_transform(X_test_preprocessed, y_test)
        X_test_preprocessed = preprocessor.transform(X_test_preprocessed)
        
        return X_test_preprocessed
    except Exception as e:
        logging.error(e)
        raise e
    

if __name__ == "__main__":
    result = get_data_for_test()
    print(result)