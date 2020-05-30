# API
#  response POST /predict
#  receive price and vin
from flask import Flask, jsonify, request
from analytics.libs.extract_features import extract_features
from analytics.libs.predict_model import predict_model
from analytics.libs.check_inputs import check_inputs

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Example test http://127.0.0.1:5000/predict?VIN=test&price=123
    :return:
    """
    if not check_inputs(**request.values):
        raise ValueError("Wrong inputs")
    features = extract_features(**request.values)
    pred = predict_model(**features)
    result = {
        **features,
        "pred": pred
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
