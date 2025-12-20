# preprocess.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
INPUT_CSV = os.path.join(DATA_DIR, "diabetes_dataset.csv")

print("Membaca:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

if 'PatientID' in df.columns:
    df = df.drop(columns=['PatientID'])

num_cols = []
for c in ['Age','BMI','BloodPressure','Insulin','Glucose','DiabetesPedigreeFunction']:
    if c in df.columns:
        num_cols.append(c)
        df[c] = pd.to_numeric(df[c], errors='coerce')

if 'BloodPressure' in df.columns and df['BloodPressure'].dtype == object:
    def parse_bp(v):
        try:
            if isinstance(v, str) and '/' in v:
                s, d = v.split('/')
                return float(s), float(d)
        except:
            pass
        return np.nan, np.nan

    systolic = []
    diastolic = []
    for v in df['BloodPressure']:
        s, d = parse_bp(v)
        systolic.append(s)
        diastolic.append(d)
    df['systolic'] = pd.Series(systolic)
    df['diastolic'] = pd.Series(diastolic)
    df['systolic'] = df['systolic'].fillna(df['systolic'].median())
    df['diastolic'] = df['diastolic'].fillna(df['diastolic'].median())
    df = df.drop(columns=['BloodPressure'])
    num_cols = [c for c in num_cols if c != 'BloodPressure']
    num_cols += ['systolic','diastolic']

if num_cols:
    imp = SimpleImputer(strategy='median')
    df[num_cols] = imp.fit_transform(df[num_cols])

if 'systolic' in df.columns and 'diastolic' in df.columns:
    df['high_bp'] = ((df['systolic'] >= 130) | (df['diastolic'] >= 80)).astype(int)

TARGET = 'Outcome'
if TARGET not in df.columns:
    raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di {INPUT_CSV}")

y = df[TARGET]
X = df.drop(columns=[TARGET])

cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)
X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False)

print("Preprocessing selesai. Files saved to:", OUT_DIR)
