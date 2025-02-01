# Heart Disease Predictor AI

An AI-powered web application that predicts heart disease risk using machine learning. Built with Flask, XGBoost, and modern web technologies.

## Features

- Real-time heart disease risk prediction
- Interactive web interface with 3D particle effects
- Comprehensive health data analysis
- Instant results with detailed insights
- Mobile-responsive design
- Modern UI with Tailwind CSS

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **ML Model**: XGBoost
- **Animations**: Lottie, Three.js
- **Styling**: Tailwind CSS

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-predictor-ai.git
   cd heart-disease-predictor-ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   cd HeartDiseaseMLInterpretation/heart_disease_app
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
HeartDiseaseMLInterpretation/
├── heart_disease_app/
│   ├── templates/
│   │   ├── index.html
│   │   └── landing.html
│   ├── static/
│   │   └── assets/
│   └── app.py
├── models/
│   └── xgb_trained.ubj
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Heart disease dataset from UCI Machine Learning Repository
- Lottie animations from LottieFiles
- Icons from Heroicons 