from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    model_name: str = "randomforest"
    do_fine_tuning: bool = False