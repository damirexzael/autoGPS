from .check_inputs import check_inputs
from .extract_features import extract_features
from .predict_model import predict_model


def predict_form(form_values):
    if not check_inputs(**form_values):
        raise ValueError("Wrong inputs")
    features = extract_features(**form_values)
    pred = predict_model(**features)
    return {
        **features,
        "pred": pred
    }
