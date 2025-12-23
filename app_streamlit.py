import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="AiLen Health Assistant", page_icon="🤖", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'compliance_score' not in st.session_state:
    st.session_state['compliance_score'] = 0

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
                "Age": 40, "BMI": bmi, "BloodPressure": 80.0, "Insulin": 80.0,
                "Glucose": glucose, "DiabetesPedigreeFunction": 0.5,
                "high_bp": 0, "Gender_Male": 1
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
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color}
        }
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
        gauge={
            "axis": {"range": [10, 45]},
            "bar": {"color": color}
        }
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
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Umur", 10, 80, 21)
        height = st.number_input("Tinggi (cm)", 100, 250, 170)
        weight = st.number_input("Berat (kg)", 30, 200, 65)
        glucose = st.number_input("Glukosa", 50, 300, 100)
        st.subheader("Gaya Hidup")
        fcvc = st.slider("Makan Sayur (1-3)", 1, 3, 2)
        tue = st.slider("Jam Gadget (0-24)", 0, 24, 5)
        faf = st.slider("Aktivitas/Olahraga (0-3)", 0, 3, 1)
    bmi = compute_bmi(height, weight)
    cat, reco, _ = bmi_category_and_advice(bmi)
    st.session_state['compliance_score'] = compliance_score(fcvc, tue, faf)

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

    st.subheader("💬 Chat dengan AiLen")
    chat_container = st.container()
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Tanya AiLen...")
        send_btn = st.form_submit_button("Kirim")
    if send_btn and user_input:
        st.session_state['chat_history'].append({"role": "user", "message": user_input})
        reply = get_bot_resp(height, weight, glucose, fcvc, tue, faf)
        st.session_state['chat_history'].append({"role": "bot", "message": reply})
    with chat_container:
        for chat in st.session_state['chat_history']:
            if chat["role"] == "user":
                st.success(f"👤 Kamu: {chat['message']}")
            else:
                st.info(f"🤖 AiLen: {chat['message']}")

if st.session_state['logged_in']:
    dashboard()
else:
    login_page()
