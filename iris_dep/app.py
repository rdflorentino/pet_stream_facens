import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('svm_26.joblib')
flower_dict = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = flower_dict[prediction[0]]

    return render_template('index.html', prediction_text='The flower is Iris {}'.format(output))


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(str(output))


if __name__ == "__main__":
    app.run(debug=True)
