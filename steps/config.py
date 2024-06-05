from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    model_name: str = "xgboost"
    do_fine_tuning: bool = False