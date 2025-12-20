from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import traceback
import re

app = FastAPI(title="AiLen - Personal Health Assistant")

# --------- Load Model Artifact ---------
MODEL_PATH = "models/diabetes_rf.joblib"
model = None
scaler = None
features = None
load_error = None

if os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        features = artifact.get("features")
    except Exception as e:
        load_error = str(e)
        with open("model_load_error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
else:
    load_error = f"Model file not found at {MODEL_PATH}"

# --------- Request Models ---------
class ChatRequest(BaseModel):
    message: str

# --------- Helper Functions ---------
def calculate_bmi_and_category(height_cm: float, weight_kg: float):
    height_m = height_cm / 100.0
    if height_m <= 0:
        return None, "Invalid"

    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        cat = "Underweight"
        reco = "Kamu butuh lebih banyak kalori sehat dan latihan kekuatan otot."
    elif bmi < 25:
        cat = "Normal"
        reco = "Pertahankan pola makan seimbang dan olahraga rutin ya!"
    elif bmi < 30:
        cat = "Overweight"
        reco = "Kurangi kalori dan tambah aktivitas fisik intensitas sedang."
    else:
        cat = "Obesitas"
        reco = "Fokus pada defisit kalori, latihan kardio, dan pola makan teratur."

    return bmi, (cat, reco)

def model_predict_from_bmi(age, bmi, blood_pressure, insulin, glucose, dpf, high_bp, gender_male):
    if model is None or scaler is None or features is None:
        return None, "model_not_ready"

    data = {
        "Age": age,
        "BMI": bmi,
        "BloodPressure": blood_pressure,
        "Insulin": insulin,
        "Glucose": glucose,
        "DiabetesPedigreeFunction": dpf,
        "high_bp": high_bp,
        "Gender_Male": gender_male,
    }

    df = pd.DataFrame([data])
    df_aligned = df.reindex(columns=features, fill_value=0)

    try:
        Xs = scaler.transform(df_aligned)
        score = float(model.predict_proba(Xs)[:, 1][0])
        return score, None
    except Exception as e:
        return None, str(e)

# --------- Endpoint ---------
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    msg = req.message.lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", msg)
    if len(numbers) < 3:
        return {"response": "Mohon sertakan tinggi, berat, dan glukosa dalam pesan Anda."}

    try:
        height = float(numbers[0])
        weight = float(numbers[1])
        glucose = float(numbers[2])
    except:
        return {"response": "Format angka tidak valid. Gunakan angka seperti: tinggi 160 berat 70 glukosa 110"}

    bmi, (cat, reco) = calculate_bmi_and_category(height, weight)
    score, err = model_predict_from_bmi(
        age=40, bmi=bmi, blood_pressure=80.0, insulin=80.0,
        glucose=glucose, dpf=0.5, high_bp=0, gender_male=1
    )

    if err == "model_not_ready":
        return {"response": "Model belum siap. Mohon pastikan file model tersedia."}

    food = "Pertahankan pola makan seimbang."
    sport = "Lanjutkan aktivitas rutin seperti yoga atau jalan pagi."

    if bmi >= 30:
        food = "Ganti nasi putih dengan nasi merah/ubi, kurangi gorengan."
        sport = "Jalan kaki 20–30 menit setiap hari."
    elif bmi >= 25:
        food = "Kurangi minuman manis, perbanyak sayur & protein."
        sport = "Stretching ringan dan naik tangga."

    response = (
        f"Hai, aku AiLen! BMI kamu {bmi:.1f} ({cat}). {reco} "
        f"Risiko diabetes sekitar {score:.2f}. "
        f"Untuk makanan: {food}. Untuk aktivitas: {sport}."
    )

    return {"response": response}
