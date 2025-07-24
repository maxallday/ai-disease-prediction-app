
# Multi-Disease Risk Predictor

[ Live App](https://ai-disease-prediction-app.streamlit.app)

An AI-powered web application that predicts the risk of **Diabetes** or **Heart Disease** using user-provided health metrics.

---

##  Features

- **Dual Prediction Models** — Supports risk prediction for both Diabetes and Heart Disease
- **Interactive Visualization** — Gauge chart shows model confidence visually
- **Validated Inputs** — Ensures user entries follow safe, expected medical ranges
- **Prediction History** — Automatically logs each prediction with time and condition
- **Data Export** — Download your prediction history as a CSV file

---

##  Screenshots

###  Input Form
![Input Form](screenshots/inputs.png)

###  Prediction Results
![Results Display](screenshots/results.png)

###  History View
![Prediction History](screenshots/history.png)

---

##  How It Works

1. **Select Disease** — Choose either Diabetes or Heart Disease
2. **Enter Metrics** — Input your current health measurements
3. **Get Prediction** — Instantly receive a risk score with confidence level
4. **View Recommendations** — Get AI-generated health advice

---

##  Input Parameters

###  Diabetes Prediction

| Parameter            | Description                               | Range   |
|----------------------|-------------------------------------------|---------|
| Pregnancies          | Number of pregnancies                     | 0–20    |
| Glucose Level        | Blood glucose concentration (mg/dL)       | 0–200   |
| Blood Pressure       | Diastolic blood pressure (mm Hg)          | 0–180   |
| Skin Thickness       | Triceps skin fold thickness (mm)          | 0–100   |
| Insulin              | 2-Hour serum insulin (mu U/ml)            | 0–300   |
| BMI                  | Body Mass Index                           | 0–60    |
| Diabetes Pedigree    | Family history likelihood                 | 0–2.5   |
| Age                  | Age in years                              | 1–120   |

###  Heart Disease Prediction

| Parameter            | Description                               | Range / Options        |
|----------------------|-------------------------------------------|------------------------|
| Age                  | Age in years                              | 20–100                 |
| Sex                  | Biological sex                            | Male / Female          |
| Chest Pain Type      | Type of chest pain                        | ATA / NAP / ASY / TA   |
| Resting BP           | Resting blood pressure (mm Hg)            | 80–200                 |
| Cholesterol          | Serum cholesterol (mg/dL)                 | 100–400                |
| Fasting BS           | Fasting blood sugar >120 mg/dl?           | Yes / No               |
| Resting ECG          | ECG results                               | Normal / ST / LVH      |
| Max HR               | Maximum heart rate                        | 60–220                 |
| Exercise Angina      | Exercise-induced angina?                  | Yes / No               |
| Oldpeak              | ST depression from exercise               | 0.0–6.0                |
| ST Slope             | Slope of peak ST segment                  | Up / Flat / Down       |

---

##  Output Interpretation

- **Low Risk** — Green display with positive suggestions
- **High Risk** — Red warning with urgent health guidance
- **Confidence Level** — Gauge from 0%–100% representing model certainty

```python
# Core prediction logic (simplified)
prediction = model.predict(user_input)[0]
confidence = model.predict_proba(user_input)[0][prediction] * 100
```

---

##  Project Structure

```
├── venv/                # Virtual environment
├── data/                # Raw datasets
│   ├── heart.csv
│   └── pima_diabetes_data.csv
├── models/              # Trained machine learning models
├── notebooks/           # Model training and experiments (Jupyter)
│   ├── diabetes-prediction.ipynb
│   └── heart-disease-prediction.ipynb
├── screenshots/         # App interface screenshots
│   ├── inputs.png
│   ├── results.png
│   └── history.png
├── app.py               # Streamlit application
└── README.md            # This file
```

---

##  Disclaimer

This app is **not a substitute for medical diagnosis**. Please consult a licensed healthcare provider for any health concerns.

---

© 2025 HealthGuard AI — All rights reserved.
