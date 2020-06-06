from flask import Flask, jsonify, request, render_template
from analytics.libs import MachineLearning

app = Flask(__name__, template_folder='templates')
ml = MachineLearning('test_pipeline')
ml.load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Example http://127.0.0.1:5000/
    """
    output = dict()
    if request.method == 'POST':
        output = ml.predict_dict(dict(request.values))
    return render_template(
        "index.html",
        method=request.method,
        predict=output,
        features=ml.features,
        values=request.values
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """ Example: http://127.0.0.1:5000/predict?asd=asd&a1=qwe123 """
    # output = predict_form(request.values)
    return jsonify(3.6)


if __name__ == '__main__':
    app.run(debug=True)
