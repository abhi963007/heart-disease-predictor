import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, render_template, jsonify, send_from_directory

# Add the parent directory to Python path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost_model_pipeline import Model

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'xgb_trained.ubj')
loaded_model = Model.load(MODEL_PATH)

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/landing.html')
def landing():
    return render_template('landing.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    
    # Rest of your prediction code...
    data = request.get_json()
    
    # Map categorical variables
    sex_mapping = {'Male': 1, 'Female': 0}
    cp_mapping = {
        'Typical Angina': 0,
        'Atypical Angina': 1,
        'Non-Anginal Pain': 2,
        'Asymptomatic': 3
    }
    restecg_mapping = {
        'Normal': 0,
        'ST-T Wave Abnormality': 1,
        'Left Ventricular Hypertrophy': 2
    }
    slope_mapping = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }
    thal_mapping = {
        'Normal': 0,
        'Fixed Defect': 1,
        'Reversible Defect': 2
    }

    try:
        # Validate and process input data
        features = np.array([
            float(data['age']),
            sex_mapping[data['sex']],
            cp_mapping[data['cp']],
            float(data['trestbps']),
            float(data['chol']),
            1 if data['fbs'] == 'Yes' else 0,
            restecg_mapping[data['restecg']],
            float(data['thalach']),
            1 if data['exang'] == 'Yes' else 0,
            float(data['oldpeak']),
            slope_mapping[data['slope']],
            float(data['ca']),
            thal_mapping[data['thal']]
        ]).reshape(1, -1)

        # Input validation
        if not (80 <= features[0][3] <= 200):  # Blood pressure
            return jsonify({'error': 'Blood pressure should be between 80 and 200 mmHg'})
        if not (100 <= features[0][4] <= 600):  # Cholesterol
            return jsonify({'error': 'Cholesterol should be between 100 and 600 mg/dl'})
        if not (60 <= features[0][7] <= 220):  # Max heart rate
            return jsonify({'error': 'Maximum heart rate should be between 60 and 220 bpm'})

        prediction = loaded_model.predict(features)[0]
        risk_level = 'High Risk' if prediction == 1 else 'Low Risk'
        
        message = "Please note that this is a preliminary assessment and should be verified by a healthcare professional."
        if risk_level == 'High Risk':
            message = "Based on the provided information, you may be at higher risk. It is strongly recommended to consult with a healthcare provider for a thorough evaluation. " + message
        else:
            message = "Based on the provided information, you appear to be at lower risk. However, maintaining a healthy lifestyle is still important. " + message

        return jsonify({
            'risk_level': risk_level,
            'message': message
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True) 