# Load trained model and perform disease prediction based on symptoms

import pickle
import os
import numpy as np
from utils.severity import calculate_severity, load_severity

model = pickle.load(open("model/model.pkl", "rb"))
label_encoder = pickle.load(open("model/label_encoder.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

BASE = os.path.dirname(os.path.dirname(__file__))
severity_dict = load_severity(os.path.join(BASE, "data", "Symptom-severity.csv"))

# Convert symptoms into binary feature vector and predict diseases
def predict_disease(symptoms):
    x = [0]*len(columns)
    for s in symptoms:
        if s in columns:
            x[columns.index(s)] = 1

    x = np.array(x).reshape(1,-1)
    probs = model.predict_proba(x)[0]
    probs = probs / probs.sum()
    idx = probs.argsort()[-3:][::-1]

    model_classes = model.classes_.astype(int)
    diseases = label_encoder.inverse_transform(model_classes)

    # Predict probabilities for all diseases
    results = [{"disease": diseases[i], "probability": float(probs[i])} for i in idx]

    score, level = calculate_severity(symptoms, severity_dict)
    confidence_score = float(max(probs))

    return results, score, level, confidence_score
    

