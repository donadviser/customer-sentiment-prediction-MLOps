from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract base defingin strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Preprocess data

        Args:
            data (pd.DataFrame): Data to be preprocessed

        Returns:
            Union[pd.DataFrame, pd.Series]: Preprocessed data
        """
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 
                         'order_delivered_carrier_date', 'order_delivered_customer_date', 
                         'order_estimated_delivery_date', 'shipping_limit_date'
                         ]
        
        id_cols = ['order_id', 'customer_id', 'customer_unique_id', 
                   'order_item_id', 'product_id', 'seller_id'
                   ]
        
        redundant_cols = ['product_name_lenght', 'product_description_lenght', 
                          'product_photos_qty', 'review_score', 'order_delivered_customer_date',
                          'order_approved_at', 'order_estimated_delivery_date', 'order_purchase_timestamp',
                          'product_length_cm', 'product_height_cm', 'product_width_cm',
                          'price', 'freight_value'
                          ]

        try:
            data = (data
             .drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
                )
            .assign(product_weight_g = lambda x: x["product_weight_g"].fillna(x["product_weight_g"].median()),
                    product_length_cm = lambda x: x["product_length_cm"].fillna(x["product_length_cm"].median()),
                    product_height_cm = lambda x: x["product_height_cm"].fillna(x["product_height_cm"].median()),
                    product_width_cm = lambda x: x["product_width_cm"].fillna(x["product_width_cm"].median()),
                    review_comment_message = lambda x:x["review_comment_message"].fillna("No review")
                    )
            .select_dtypes(include=[np.number])
            .drop(["customer_zip_code_prefix", "order_item_id"], axis=1)
            )
            logging.info(data.head())
            print(data.columns)
            return data
        except Exception as e:
            raise CustomException(e, "Errorr while processing data")
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data

        Args:
            data (pd.DataFrame): Data to be divided

        Returns:
            Union[pd.DataFrame, pd.Series]: Divided data
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, "Errorr while dividing data")
        
class DataCleaning:
    """
    Class for cleaning data: Clean, Preprocess and Divide the data
    """
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Clean, Preprocess and Divide the data

        Returns:
            Union[pd.DataFrame, pd.Series]: Clean, Preprocessed and Divided data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            raise CustomException(e, "Errorr while cleaning, preprocessing and dividing data")