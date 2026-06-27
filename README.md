# 🏥 AI-Powered Multi-Disease Health Risk Analyzer

> Simultaneously predicts risk for **Diabetes, Heart Disease, Liver Disease, and Kidney Failure** using Gradient Boosting ML models with SHAP-style explainability.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0-green) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-88%25+-teal)

---

## 🚀 Features

- **4 Disease Models** — Diabetes, Heart Disease, Liver Disease, Kidney Failure
- **Gradient Boosting Classifier** — 88–90% accuracy, 0.92+ AUC on all models
- **SHAP-style Explainability** — Shows which factors drive each prediction
- **Personalized Recommendations** — Actionable lifestyle advice based on inputs
- **Clean Web UI** — 4-tab interface: Input → Results → Explanation → Recommendations
- **REST API** — Easy to extend or connect to a mobile app

---

## 📁 Project Structure

```
health-analyzer/
├── app.py               # Flask backend + prediction API
├── train_models.py      # Train/retrain all 4 ML models
├── requirements.txt     # Python dependencies
├── models/              # Saved .pkl model files
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   ├── liver_model.pkl
│   └── kidney_model.pkl
├── data/                # Training datasets (CSV)
└── frontend/
    └── index.html       # Full web UI (vanilla HTML/CSS/JS)
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/subhramukul/health-risk-analyzer.git
cd health-risk-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train models (first time only)
```bash
python train_models.py
```

### 4. Run the app
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🌐 Deploy on Render (Free)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt && python train_models.py`
   - **Start Command:** `gunicorn app:app`
5. Done — live public URL in 2 minutes!

---

## 📊 Model Performance

| Disease       | Accuracy | AUC   |
|---------------|----------|-------|
| Diabetes      | 88.0%    | 0.944 |
| Heart Disease | 85.5%    | 0.924 |
| Liver Disease | 90.5%    | 0.964 |
| Kidney Failure| 83.5%    | 0.920 |

---

## 🔌 API Reference

### `POST /predict`
```json
{
  "age": 45,
  "weight": 80,
  "height": 175,
  "bp": 130,
  "glucose": 110,
  "cholesterol": 210,
  "heartrate": 75,
  "smoking": "Former",
  "activity": "Light",
  "alcohol": "Occasional",
  "family": "Diabetes"
}
```

**Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "disease": "Diabetes",
      "probability": 62.3,
      "level": "high",
      "top_factors": [...]
    }
  ],
  "recommendations": [...],
  "summary": { "avg_risk": 48.2, "high_count": 1, "bmi": 26.1 }
}
```

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **ML:** Scikit-learn (Gradient Boosting), Joblib, NumPy, Pandas
- **Explainability:** SHAP-style feature perturbation analysis
- **Frontend:** Vanilla HTML/CSS/JS (no framework needed)
- **Deployment:** Gunicorn + Render

---

## 👤 Author

**Subhramukul Payra**  
B.Tech CSE (AI/ML) — KIIT University  
GitHub: [@subhramukul](https://github.com/subhramukul)  
LinkedIn: [subhramukul-payra](https://linkedin.com/in/subhramukul-payra-b4180527a)

---

> ⚠️ **Disclaimer:** This tool is for educational and portfolio demonstration purposes only. It is not a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.
