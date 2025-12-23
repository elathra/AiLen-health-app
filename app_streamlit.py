import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(page_title="AiLen Health Assistant", page_icon="🤖", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'compliance_score' not in st.session_state:
    st.session_state['compliance_score'] = 0
if 'sidebar_values' not in st.session_state:
    st.session_state['sidebar_values'] = {
        "gender": "Female",
        "age": 21,
        "height": 170,
        "weight": 65,
        "glucose": 100,
        "fcvc": 2,
        "tue": 5,
        "faf": 1
    }

model, scaler, features = None, None, None
MODEL_PATH = "models/diabetes_rf.joblib"
if os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        features = artifact.get("features")
    except:
        pass

def login_page():
    st.title("🔐 Login ke AiLen")
    with st.form("login_form"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        submit = st.form_submit_button("Masuk")
        if submit:
            if user and pw:
                st.session_state['logged_in'] = True
                st.session_state['username'] = user
                if not st.session_state['chat_history']:
                    st.session_state['chat_history'].append({
                        "role": "bot",
                        "message": f"Halo {user}! Aku AiLen. Isi profilmu di sidebar, lalu tanya apa saja."
                    })
                st.success("Login Berhasil!")
                st.stop()
            else:
                st.error("Isi username dan password.")

def compute_bmi(height, weight):
    try:
        return weight / ((height / 100) ** 2)
    except:
        return None

def bmi_category_and_advice(bmi):
    if bmi is None:
        return "Invalid", "Periksa input tinggi/berat.", ("Periksa input", "Periksa input")
    if bmi < 18.5:
        return "Underweight", "Tambah kalori sehat dan latihan kekuatan otot.", ("Tambah porsi protein, sayur, dan karbo kompleks.", "Latihan kekuatan ringan 2–3x/minggu.")
    if bmi < 25:
        return "Normal", "Pertahankan pola makan seimbang dan olahraga rutin.", ("Porsi seimbang karbo-protein-sayur.", "Jalan cepat/yoga 3–4x/minggu.")
    if bmi < 30:
        return "Overweight", "Kurangi kalori, tambah aktivitas sedang.", ("Kurangi minuman manis dan gorengan.", "Naik tangga, jalan 30 menit/hari.")
    return "Obesitas", "Fokus pada defisit kalori dan latihan kardio.", ("Ganti nasi putih dengan sumber karbo tinggi serat.", "Kardio ringan 30–40 menit, 4–5x/minggu.")

def risk_score(glucose, bmi):
    score_txt = "Tidak tersedia"
    if model and scaler and features and bmi is not None:
        try:
            data = {
                "Age": 40,
                "BMI": bmi,
                "BloodPressure": 80.0,
                "Insulin": 80.0,
                "Glucose": glucose,
                "DiabetesPedigreeFunction": 0.5,
                "high_bp": 0,
                "Gender_Male": 1,
                "family_history_with_overweight": 1 if st.session_state['sidebar_values']["family_history_with_overweight"]=="Yes" else 0
            }
            df = pd.DataFrame([data])
            df_aligned = df.reindex(columns=features, fill_value=0)
            Xs = scaler.transform(df_aligned)
            score = float(model.predict_proba(Xs)[:, 1][0])
            score_txt = f"{score:.2f}"
        except:
            pass
    return score_txt


def compliance_score(fcvc, tue, faf):
    veg_score = (fcvc - 1) / 2
    gadget_score = max(0, (24 - tue) / 24)
    activity_score = faf / 3
    score = (0.4 * veg_score + 0.3 * gadget_score + 0.3 * activity_score) * 100
    return round(score)

def compliance_gauge(score):
    color = "#E74C3C" if score < 40 else ("#F1C40F" if score < 70 else "#2ECC71")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Skor Kepatuhan"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}}
    ))
    st.plotly_chart(fig, use_container_width=True)

def draw_bmi_gauge(bmi):
    if bmi is None:
        st.metric("BMI", "—")
        return
    if bmi < 18.5:
        color = "#33C4FF"
    elif bmi < 25:
        color = "#2ECC71"
    elif bmi < 30:
        color = "#F1C40F"
    else:
        color = "#E74C3C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bmi,
        title={"text": "BMI"},
        gauge={"axis": {"range": [10, 45]}, "bar": {"color": color}}
    ))
    st.plotly_chart(fig, use_container_width=True)

def draw_macro(cat):
    if "Obesitas" in cat:
        values, title = [20, 50, 30], "Low Carb"
    elif "Overweight" in cat:
        values, title = [40, 40, 20], "Kurangi Lemak"
    elif "Underweight" in cat:
        values, title = [55, 25, 20], "Kalori Naik"
    else:
        values, title = [50, 30, 20], "Diet Normal"
    labels = ["Karbo", "Protein", "Lemak"]
    colors = ["#F4D03F", "#58D68D", "#5DADE2"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors, sort=False)])
    fig.update_layout(title_text=f"Target Nutrisi ({title})")
    st.plotly_chart(fig, use_container_width=True)

def get_bot_resp(height, weight, glucose, fcvc, tue, faf):
    bmi = compute_bmi(height, weight)
    cat, reco, (food, sport) = bmi_category_and_advice(bmi)
    score_txt = risk_score(glucose, bmi)
    lifestyle = []
    if fcvc < 2:
        lifestyle.append("Tingkatkan porsi sayur.")
    if tue > 8:
        lifestyle.append("Kurangi waktu gadget.")
    if faf == 0:
        lifestyle.append("Mulai olahraga ringan.")
    lifestyle_txt = " ".join(lifestyle)
    bmi_txt = f"{bmi:.1f}" if bmi is not None else "—"
    return f"BMI {bmi_txt} ({cat}). {reco} Risiko diabetes: {score_txt}. Makanan: {food}. Aktivitas: {sport}. {lifestyle_txt}"

def dashboard():
    with st.sidebar:
        st.title(f"👤 Hi, {st.session_state['username']}")
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.stop()

        st.header("Profil Fisik")
        gender = st.selectbox("Gender", ["Female", "Male"], 
                              index=0 if st.session_state['sidebar_values']['gender']=="Female" else 1)
        age = st.slider("Umur", 10, 80, st.session_state['sidebar_values']['age'])
        height = st.number_input("Tinggi (cm)", 100, 250, st.session_state['sidebar_values']['height'])
        weight = st.number_input("Berat (kg)", 30, 200, st.session_state['sidebar_values']['weight'])
        glucose = st.number_input("Glukosa", 50, 300, st.session_state['sidebar_values']['glucose'])
        family_history = st.radio(
            "Riwayat Diabetes Turunan + Overweight",
            ["No", "Yes"],
            index=0 if st.session_state['sidebar_values'].get("family_history_with_overweight","No")=="No" else 1
        )

        st.subheader("Gaya Hidup")
        fcvc = st.slider("Makan Sayur (1-3)", 1, 3, st.session_state['sidebar_values']['fcvc'])
        tue = st.slider("Jam Gadget (0-24)", 0, 24, st.session_state['sidebar_values']['tue'])
        faf = st.slider("Aktivitas/Olahraga (0-3)", 0, 3, st.session_state['sidebar_values']['faf'])

        st.session_state['sidebar_values'] = {
            "gender": gender,
            "age": age,
            "height": height,
            "weight": weight,
            "glucose": glucose,
            "fcvc": fcvc,
            "tue": tue,
            "faf": faf,
            "family_history_with_overweight": family_history
        }

    bmi = compute_bmi(st.session_state['sidebar_values']['height'], st.session_state['sidebar_values']['weight'])
    cat, reco, _ = bmi_category_and_advice(bmi)
    st.session_state['compliance_score'] = compliance_score(
        st.session_state['sidebar_values']['fcvc'],
        st.session_state['sidebar_values']['tue'],
        st.session_state['sidebar_values']['faf']
    )

    st.title("🧬 AiLen Health Dashboard")
    col_top = st.columns([1, 1, 1])
    with col_top[0]:
        st.subheader("BMI")
        draw_bmi_gauge(bmi)
        st.info(f"Kategori BMI: {f'{bmi:.1f}' if bmi is not None else '—'}")
    with col_top[1]:
        st.subheader("Skor Kepatuhan")
        compliance_gauge(st.session_state['compliance_score'])
        st.info(f"Skor: {st.session_state['compliance_score']}/100")
    with col_top[2]:
        st.subheader("Target Nutrisi")
        draw_macro(cat)


def chatbot_tab():
    st.subheader("💬 Chat dengan AiLen")
    chat_container = st.container()
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Tanya AiLen...")
        send_btn = st.form_submit_button("Kirim")
    if send_btn and user_input:
        v = st.session_state['sidebar_values']
        reply = get_bot_resp(v['height'], v['weight'], v['glucose'], v['fcvc'], v['tue'], v['faf'])
        st.session_state['chat_history'].append({"role": "user", "message": user_input})
        st.session_state['chat_history'].append({"role": "bot", "message": reply})
    with chat_container:
        for chat in st.session_state['chat_history']:
            if chat["role"] == "user":
                st.success(f"👤 Kamu: {chat['message']}")
            else:
                st.info(f"🤖 AiLen: {chat['message']}")

def evaluasi_ai():
    st.header("📊 Evaluasi AI vs Ground Truth")
    uploaded = st.file_uploader("Upload compare_results.csv", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df["Error"] = df["MyAI"] - df["GroundTruth"]
        df["AbsError"] = df["Error"].abs()
        mae = df["AbsError"].mean()
        rmse = np.sqrt((df["Error"]**2).mean())
        mape = (df["AbsError"] / df["GroundTruth"].replace(0, np.nan)).mean() * 100
        accuracy = 100 - mape if np.isfinite(mape) else np.nan
        speed_avg = df["inference_ms"].mean() if "inference_ms" in df.columns else np.nan
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi (MAPE)", f"{accuracy:.2f}%")
        col2.metric("MAE", f"{mae:.3f}")
        col3.metric("RMSE", f"{rmse:.3f}")
        col4.metric("Rata-rata kecepatan", f"{speed_avg:.2f} ms" if not np.isnan(speed_avg) else "N/A")
        fig_line = go.Figure()
        x = df["Sample"] if "Sample" in df.columns else list(range(len(df)))
        fig_line.add_trace(go.Scatter(x=x, y=df["GroundTruth"], mode="lines+markers", name="Ground Truth"))
        fig_line.add_trace(go.Scatter(x=x, y=df["MyAI"], mode="lines+markers", name="My AI"))
        fig_line.update_layout(title="Prediksi vs Ground Truth", xaxis_title="Sample", yaxis_title="Nilai")
        st.plotly_chart(fig_line, use_container_width=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=x, y=df["Error"], name="Error"))
        fig_bar.update_layout(title="Error Prediksi", xaxis_title="Sample", yaxis_title="Error")
        st.plotly_chart(fig_bar, use_container_width=True)
        if "inference_ms" in df.columns:
            fig_speed = go.Figure()
            fig_speed.add_trace(go.Scatter(x=x, y=df["inference_ms"], mode="lines+markers", name="Kecepatan (ms)"))
            fig_speed.update_layout(title="Kecepatan Inference per Sampel", xaxis_title="Sample", yaxis_title="ms")
            st.plotly_chart(fig_speed, use_container_width=True)
        st.subheader("📋 Data Ringkas")
        st.dataframe(df)
    else:
        st.info("Upload file compare_results.csv untuk melihat evaluasi.")

def main_app():
    tabs = st.tabs(["Dashboard", "Chatbot", "Evaluasi AI"])
    with tabs[0]:
        dashboard()
    with tabs[1]:
        chatbot_tab()
    with tabs[2]:
        evaluasi_ai()

if st.session_state['logged_in']:
    main_app()
else:
    login_page()
