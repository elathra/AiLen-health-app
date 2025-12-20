# train_baseline.py
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train_scaled, y_train)

proba = model.predict_proba(X_test_scaled)[:, 1]
pred = (proba >= 0.5).astype(int)

auc = roc_auc_score(y_test, proba)
print(f"AUC: {auc:.4f}")
print("Classification report:")
print(classification_report(y_test, pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))

artifact = {"model": model, "scaler": scaler, "features": list(X_train.columns)}
out_path = os.path.join(MODEL_DIR, "diabetes_rf.joblib")
joblib.dump(artifact, out_path)
print("Model dan artefak disimpan ke:", out_path)
