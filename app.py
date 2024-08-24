# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model
with open('linear_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input value from the form
        feature_value = float(request.form['feature'])

        # Make a prediction using the loaded model
        prediction = model.predict([[feature_value]])

        return render_template('index.html', prediction=f'The predicted price is: {prediction[0]:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
