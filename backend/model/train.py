# Train and tune a Random Forest model using GridSearchCV
# Saves trained model, label encoder, and feature columns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.preprocessing import load_data, split_data
from sklearn.preprocessing import LabelEncoder
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
DATA_FILE = os.path.join(BASE_DIR, "../data/dataset.csv")

# Load dataset and split into features (X) and labels (y)
df = load_data(DATA_FILE)
X, y = split_data(df)

# Encode disease labels into numeric format
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# Random Forest
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

# Perform hyperparameter tuning for Random Forest
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=3,
    verbose=1,
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# Evaluate model accuracy on test data
rf_acc = accuracy_score(y_test, rf_best.predict(X_test))
print(f"RF Accuracy: {rf_acc}")

with open(os.path.join(MODEL_DIR, "metrics.txt"), "w") as f:
    f.write(str(rf_acc))

# Save trained model and preprocessing artifacts
os.makedirs(MODEL_DIR, exist_ok=True)

pickle.dump(rf_best, open(os.path.join(MODEL_DIR, "model.pkl"), "wb"))
pickle.dump(label_encoder, open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb"))
pickle.dump(X.columns.tolist(), open(os.path.join(MODEL_DIR, "columns.pkl"), "wb"))

print("✔ Best model saved!")