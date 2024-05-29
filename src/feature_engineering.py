import pandas as pd
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

from src.logger import logging
from src.exceptions import CustomException

class DateTimeConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to convert specified columns to datetime.
    """
    def __init__(self, datetime_cols: list) -> None:
        self.datetime_cols = datetime_cols
        
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DateTimeConverter':
        return self
    
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = (X
                .assign(**{col: pd.to_datetime(df[col], errors='coerce') for col in self.datetime_cols})
            )
            return X
        except Exception as e:
            raise CustomException(e, f"Error in DateTimeConverter")
        

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer to engineer new features from the dataset.
    """
    def fit(self, X: pd.DataFrame, y: None = None) -> 'FeatureEngineer':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X = (X 
                 # Calculate new features
                .assign(
                     time_to_delivery=lambda x: (x['order_delivered_customer_date'] - x['order_approved_at']).dt.days,
                     order_processing_time=lambda x: (x['order_approved_at'] - x['order_purchase_timestamp']).dt.days,
                     estimated_vs_actual_shipping=lambda x: (x['order_estimated_delivery_date'] - x['order_delivered_customer_date']).dt.days,
                     product_volume_m3=lambda x: (x['product_length_cm'] * x['product_width_cm'] * x['product_height_cm']) / 1000000,
                     satisfaction=lambda x: (x['review_score'] >= 4).astype(int),
                     order_value=lambda x: x['price'] + x['freight_value'],
                     late_delivery=lambda x: (x['order_delivered_customer_date'] > x['order_estimated_delivery_date']).astype(int)
                    )
          
                # Create seasonal features from order_purchase_timestamp
                .assign(
                        order_month=lambda x: x['order_purchase_timestamp'].dt.month,
                        order_day=lambda x: x['order_purchase_timestamp'].dt.dayofweek,
                        order_hour=lambda x: x['order_purchase_timestamp'].dt.hour
                    )
                )
            return X
        except Exception as e:
            raise CustomException(e, f"Error in FeatureEngineer")
        
        
class DropRedundantColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to drop redundant columns from the dataset.
    """
    def __init__(self, redundant_cols: list) -> None:
        self.redundant_cols = redundant_cols
    
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropRedundantColumns':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.drop(columns=self.redundant_cols)
        except Exception as e:
            raise CustomException(e, f"Error in DropRedundantColumns")
        

class DropMissingThreshold(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns with missing values above a certain threshold.
    """
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropMissingThreshold':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.dropna(thresh=len(X) * self.threshold, axis=1)
        except Exception as e:
            raise CustomException(e, f"Error in DropMissingThreshold")
        

def create_preprocessing_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, list, list]:
    """
    Create a preprocessing pipeline and define the columns for numeric and categorical transformers.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Pipeline, list, list]: A tuple containing the preprocessing pipeline,
                                     list of numerical columns, and list of categorical columns.
    """
    try:
        redundant_cols = [
            'product_name_lenght', 'product_description_lenght', 
            'product_photos_qty', 'review_score', 'order_delivered_customer_date',
            'order_approved_at', 'order_estimated_delivery_date', 'order_purchase_timestamp',
            'product_length_cm', 'product_height_cm', 'product_width_cm',
            'price', 'freight_value'
        ]

        numerical_cols = ['payment_value', 'order_value', 'product_weight_g', 'product_volume_m3', 'order_item_id', 'estimated_vs_actual_shipping',
                  'time_to_delivery', 'order_processing_time', 'payment_installments','customer_zip_code_prefix']
        categorical_cols = ['order_month', 'order_day', 'order_hour','late_delivery']
        datetime_cols = ['order_delivered_customer_date', 'order_approved_at', 'order_estimated_delivery_date', 'order_purchase_timestamp']

        pipeline = Pipeline([
            ('drop_missing_threshold', DropMissingThreshold(threshold=0.5)),
            ('convert_datetime', DateTimeConverter(datetime_cols=datetime_cols)),
            ('feature_engineering', FeatureEngineer()),
            ('drop_redundant', DropRedundantColumns(redundant_cols=redundant_cols)),
            ('drop_na', FunctionTransformer(lambda x: x.dropna(), validate=False))
        ])

        logging.info("Preprocessing pipeline created successfully.")
        return pipeline, numerical_cols, categorical_cols
    except Exception as e:
        logging.error(f"Error in create_preprocessing_pipeline: {e}")
        raise CustomException(f"Error in create_preprocessing_pipeline: {e}")

def preprocess_and_split_data(df: pd.DataFrame, target_col: str = 'satisfaction') -> Tuple:
    """
    Preprocess the data and split into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.

    Returns:
        Tuple: Preprocessed training and testing data, and the preprocessor.
    """
    try:
        pipeline, numerical_cols, categorical_cols = create_preprocessing_pipeline(df)
        df_clean = pipeline.fit_transform(df)

        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols),
            ])

        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        logging.info("Data preprocessed and split into training and testing sets successfully.")
        return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor
    except Exception as e:
        raise CustomException(e, f"Error in preprocess_and_split_data")
    
if __name__ == "__main__":
    df = pd.read_csv('/Users/don/github-projects/pre-purchase-sentiment-prediction-MLOps/data/olist_customers_dataset.csv')
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split_data(df)
    joblib.dump(preprocessor, 'artefacts/preprocessor.joblib')
    joblib.dump(X_train, 'artefacts/X_train.joblib')
    joblib.dump(X_test, 'artefacts/X_test.joblib')
    joblib.dump(y_train, 'artefacts/y_train.joblib')
    joblib.dump(y_test, 'artefacts/y_test.joblib')
    print(f"X_train.shape: {X_train.shape}")
    print(f"X_test.shape: {X_test.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"y_test.shape: {y_test.shape}")
    print(f"preprocessor: {preprocessor}")