"""
train_models.py — Run this once to generate all ML models.
Usage: python train_models.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib, json, os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

FEATURES = ['age','bmi','glucose','blood_pressure','smoking',
            'physical_activity','family_history','cholesterol','alcohol']
np.random.seed(42)
N = 1000

def generate_dataset():
    return pd.DataFrame({
        'age': np.random.randint(20,80,N),
        'bmi': np.round(np.random.normal(27,5,N),1),
        'glucose': np.random.randint(70,200,N),
        'blood_pressure': np.random.randint(60,140,N),
        'smoking': np.random.choice([0,1,2],N,p=[0.5,0.2,0.3]),
        'physical_activity': np.random.choice([0,1,2,3],N,p=[0.3,0.3,0.25,0.15]),
        'family_history': np.random.choice([0,1],N,p=[0.7,0.3]),
        'cholesterol': np.random.randint(130,280,N),
        'alcohol': np.random.choice([0,1,2,3],N,p=[0.4,0.3,0.2,0.1])
    })

base = generate_dataset()
configs = {
    'diabetes': lambda d: (d['glucose']>126)*0.4 + (d['bmi']>30)*0.2 + (d['age']>45)*0.15 + (d['family_history']==1)*0.15 + (d['smoking']==2)*0.1,
    'heart':    lambda d: (d['blood_pressure']>130)*0.35 + (d['cholesterol']>200)*0.25 + (d['smoking']==2)*0.2 + (d['age']>50)*0.15 + (d['physical_activity']==0)*0.05,
    'liver':    lambda d: (d['alcohol']>=2)*0.45 + (d['bmi']>30)*0.2 + (d['smoking']==2)*0.15 + (d['age']>40)*0.1,
    'kidney':   lambda d: (d['blood_pressure']>130)*0.35 + (d['glucose']>140)*0.25 + (d['bmi']>30)*0.15 + (d['age']>50)*0.15 + (d['smoking']==2)*0.1,
}

print("Training models...\n")
for disease, risk_fn in configs.items():
    df = base.copy()
    risk = risk_fn(df) + np.random.normal(0, 0.1, N)
    df['target'] = (risk > 0.35).astype(int)
    df.to_csv(f'data/{disease}.csv', index=False)

    X, y = df[FEATURES], df['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)

    acc = accuracy_score(y_te, model.predict(X_te))
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    print(f"  {disease:10s} → Accuracy: {acc:.3f}  |  AUC: {auc:.3f}")

    joblib.dump(model, f'models/{disease}_model.pkl')

with open('models/features.json','w') as f:
    json.dump(FEATURES, f)

print("\nAll models saved to /models/")
