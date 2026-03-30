# Calculate severity score based on symptom weights

import pandas as pd

def load_severity(path):
    df = pd.read_csv(path)
    return dict(zip(df["Symptom"], df["weight"]))

def calculate_severity(symptoms, severity_dict):
    score = sum(severity_dict.get(s, 1) for s in symptoms)
    if score < 5:
        level = "Low"
    elif score < 10:
        level = "Medium"
    elif score < 15:
        level = "High"
    else:
        level = "Critical"
    return score, level
