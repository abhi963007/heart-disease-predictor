# Heart Disease Risk Predictor Web Application

## Overview
This web application provides a user-friendly interface to predict heart disease risk using a pre-trained XGBoost machine learning model.

## Features
- Interactive web form for inputting patient health parameters
- Real-time heart disease risk prediction
- Clear visualization of risk level (High Risk/Low Risk)

## Prerequisites
- Python 3.10+
- pip (Python package manager)

## Installation

1. Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
flask run
```

Or 

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Input Features
The application requires the following input features:
1. Age
2. Sex
3. Chest Pain Type
4. Resting Blood Pressure
5. Serum Cholesterol
6. Fasting Blood Sugar
7. Resting ECG Results
8. Maximum Heart Rate
9. Exercise Induced Angina
10. ST Depression
11. ST Segment Slope
12. Number of Major Vessels
13. Thalassemia

## Model Details
- Algorithm: XGBoost Classifier
- Training Accuracy: 95%
- Precision: 96.97%
- Recall: 84.21%

## Disclaimer
This is a predictive tool and should not replace professional medical advice. Always consult with a healthcare professional.

## License
[Specify your license here] 