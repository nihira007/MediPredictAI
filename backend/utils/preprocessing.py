# Preprocess dataset: extract symptoms and convert into feature vectors

import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

# Convert symptoms into binary matrix and extract labels
def split_data(df):
    symptom_columns = [c for c in df.columns if c != "Disease"]
    unique_symptoms = []
    seen = set()
    for _, row in df[symptom_columns].iterrows():
        for value in row.dropna().astype(str):
            symptom = value.strip()
            if symptom and symptom not in seen:
                seen.add(symptom)
                unique_symptoms.append(symptom)

    X_records = []
    for _, row in df[symptom_columns].iterrows():
        row_symptoms = {v.strip() for v in row.dropna().astype(str) if v.strip()}
        X_records.append([1 if symptom in row_symptoms else 0 for symptom in unique_symptoms])

    X = pd.DataFrame(X_records, columns=unique_symptoms)
    y = df["Disease"]
    return X, y
