# Mico Ellerich M. Comia
# Reference: https://www.kdnuggets.com/2019/01
# /build-api-machine-learning-model-using-flask.html

import numpy as np
import pandas as pd
import joblib
import json
from flask import Flask, request, redirect, url_for, flash, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
	data = request.get_json()
	prediction = svm_model.predict([np.array(list(data.values()))])
	output = {"Y": int(prediction[0])}
	return jsonify(output)


if __name__ == '__main__':
	svm_model = joblib.load("model/svm_0520-1547.sav")
	app.run(debug=True, host='0.0.0.0')
