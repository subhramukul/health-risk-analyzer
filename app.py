from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, numpy as np, pandas as pd, os

app = Flask(__name__, static_folder='frontend')
CORS(app)

FEATURES = ['age','bmi','glucose','blood_pressure','smoking','physical_activity','family_history','cholesterol','alcohol']
DISEASES = ['diabetes','heart','liver','kidney']
DISEASE_LABELS = {
    'diabetes': 'Diabetes',
    'heart': 'Heart Disease',
    'liver': 'Liver Disease',
    'kidney': 'Kidney Failure'
}

models = {}
for d in DISEASES:
    models[d] = joblib.load(f'models/{d}_model.pkl')

def compute_shap_like(model, input_df, feature_names):
    base_prob = model.predict_proba(input_df)[0][1]
    importances = []
    neutral_vals = {'age': 45, 'bmi': 25, 'glucose': 95, 'blood_pressure': 110,
                    'smoking': 0, 'physical_activity': 2, 'family_history': 0,
                    'cholesterol': 175, 'alcohol': 0}
    for feat in feature_names:
        modified = input_df.copy()
        modified[feat] = neutral_vals.get(feat, 0)
        new_prob = model.predict_proba(modified)[0][1]
        impact = round(abs(base_prob - new_prob) * 100, 2)
        importances.append({'feature': feat, 'impact': impact})
    importances.sort(key=lambda x: x['impact'], reverse=True)
    return importances

def get_recommendations(inputs, risks):
    recs = []
    bmi = inputs['bmi']
    if inputs['smoking'] == 2:
        recs.append({'icon': '🚭', 'title': 'Quit smoking', 'body': 'Smoking significantly increases risk across all 4 conditions. Quitting reduces heart disease risk by 50% within a year.'})
    if inputs['physical_activity'] <= 1:
        recs.append({'icon': '🏃', 'title': 'Increase physical activity', 'body': 'Aim for 150 min/week of moderate exercise. Lowers glucose, blood pressure, and cholesterol simultaneously.'})
    if bmi > 27:
        recs.append({'icon': '⚖️', 'title': 'Work toward healthy BMI', 'body': f'Your BMI is {bmi:.1f}. Losing 5–10% of body weight reduces diabetes and heart risk by 30–40%.'})
    if inputs['glucose'] > 100:
        recs.append({'icon': '🍚', 'title': 'Monitor blood glucose', 'body': 'Fasting glucose above 100 mg/dL is pre-diabetic range. Reduce refined carbs and sugary drinks.'})
    if inputs['blood_pressure'] > 130:
        recs.append({'icon': '🫀', 'title': 'Lower blood pressure', 'body': 'BP above 130 mmHg strains kidneys and heart. Target <2300mg sodium/day and check regularly.'})
    if inputs['alcohol'] >= 2:
        recs.append({'icon': '🍷', 'title': 'Reduce alcohol intake', 'body': 'Regular/heavy drinking is the primary driver of liver disease. Limit to 1 drink/day maximum.'})
    if inputs['cholesterol'] > 200:
        recs.append({'icon': '🥑', 'title': 'Improve lipid profile', 'body': 'Cholesterol above 200 mg/dL elevates heart risk. Add omega-3s, soluble fiber, reduce saturated fats.'})
    if not recs:
        recs.append({'icon': '✅', 'title': 'Maintain your healthy lifestyle', 'body': 'Your vitals look good! Keep up your habits and get annual checkups for early detection.'})
    return recs

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        age = float(data['age'])
        weight = float(data['weight'])
        height = float(data['height']) / 100
        bmi = round(weight / (height ** 2), 1)

        smoke_map = {'Never': 0, 'Former': 1, 'Current': 2}
        act_map = {'Sedentary': 0, 'Light': 1, 'Moderate': 2, 'Active': 3}
        alc_map = {'None': 0, 'Occasional': 1, 'Regular': 2, 'Heavy': 3}
        fam_map = {'None': 0, 'Heart disease': 1, 'Diabetes': 1, 'Both': 1}

        inputs = {
            'age': age,
            'bmi': bmi,
            'glucose': float(data['glucose']),
            'blood_pressure': float(data['bp']),
            'smoking': smoke_map.get(data['smoking'], 0),
            'physical_activity': act_map.get(data['activity'], 1),
            'family_history': fam_map.get(data['family'], 0),
            'cholesterol': float(data['cholesterol']),
            'alcohol': alc_map.get(data['alcohol'], 0)
        }

        input_arr = pd.DataFrame([[inputs[f] for f in FEATURES]], columns=FEATURES)

        results = []
        all_shap = {}
        for disease in DISEASES:
            model = models[disease]
            prob = model.predict_proba(input_arr)[0][1]
            pct = round(prob * 100, 1)
            shap_vals = compute_shap_like(model, input_arr.copy(), FEATURES)
            results.append({
                'disease': DISEASE_LABELS[disease],
                'key': disease,
                'probability': pct,
                'level': 'high' if pct >= 60 else ('medium' if pct >= 30 else 'low'),
                'top_factors': shap_vals[:5]
            })
            all_shap[disease] = shap_vals

        recs = get_recommendations(inputs, results)
        avg_risk = round(sum(r['probability'] for r in results) / len(results), 1)
        high_count = sum(1 for r in results if r['level'] == 'high')

        return jsonify({
            'success': True,
            'predictions': results,
            'recommendations': recs,
            'summary': {
                'avg_risk': avg_risk,
                'high_count': high_count,
                'bmi': bmi
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': len(models)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
