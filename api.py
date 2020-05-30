# API
#  response POST /predict
#  receive price and vin
from flask import Flask, jsonify, request
from analytics.libs.extract_features import extract_features
from analytics.libs.predict_model import predict_model

app = Flask(__name__)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print(request.form.get('asd'))
    result = {
        "saludo": "Hello World!"
    }
    return jsonify(result)


if __name__ == '__main__':
    print(extract_features(vin='adas'))
    print(predict_model(unoo='a', dos='b'))
    app.run(debug=True)
