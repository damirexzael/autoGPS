from flask import Flask, jsonify, request, render_template
from analytics.libs.predict_form import predict_form

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Example http://127.0.0.1:5000/
    """
    output = dict()
    if request.method == 'POST':
        output = predict_form(request.values)
    return render_template(
        "index.html",
        predict=output.get('pred'),
        vin=request.values.get('vin', ''),
        price=request.values.get('price', ''),
        method=request.method
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """ Example: http://127.0.0.1:5000/predict?asd=asd&a1=qwe123 """
    output = predict_form(request.values)
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
