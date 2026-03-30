# Load disease descriptions and precautions from dataset

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

desc_path = os.path.join(BASE_DIR, "data", "symptom_Description.csv")
prec_path = os.path.join(BASE_DIR, "data", "symptom_precaution.csv")

desc_df = pd.read_csv(desc_path)
prec_df = pd.read_csv(prec_path)

disease_descriptions = dict(zip(desc_df["Disease"], desc_df["Description"]))
disease_precautions = {}

for _, row in prec_df.iterrows():
    disease = row["Disease"]
    precautions = [p for p in row[1:] if pd.notna(p)]
    disease_precautions[disease] = precautions

def get_description(disease):
    return disease_descriptions.get(disease, "No description available")

def get_precautions(disease):
    return disease_precautions.get(disease, ["Consult a healthcare professional"])