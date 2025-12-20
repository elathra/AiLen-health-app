import streamlit as st
import pandas as pd
import os

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except:
    PLOTLY_OK = False

try:
    import joblib
    JOBLIB_OK = True
except:
    JOBLIB_OK = False

# Konfigurasi halaman
st.set_page_config(page_title="AiLen Health Assistant", page_icon="🤖", layout="wide")

# Session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Load model
model, scaler, features = None, None, None
MODEL_PATH = "models/diabetes_rf.joblib"
if JOBLIB_OK and os.path.exists(MODEL_PATH):
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        features = artifact.get("features")
    except:
        pass

# Login page
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

# Chatbot response
def get_bot_resp(height, weight, glucose, fcvc, tue, faf):
    try:
        bmi = weight / ((height / 100) ** 2)
    except:
        bmi = None

    if bmi is None:
        cat, reco = "Invalid", "Periksa input tinggi/berat."
    elif bmi < 18.5:
        cat, reco = "Underweight", "Tambah kalori sehat dan latihan kekuatan otot."
    elif bmi < 25:
        cat, reco = "Normal", "Pertahankan pola makan seimbang dan olahraga rutin."
    elif bmi < 30:
        cat, reco = "Overweight", "Kurangi kalori, tambah aktivitas sedang."
    else:
        cat, reco = "Obesitas", "Fokus pada defisit kalori dan latihan kardio."

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

    if bmi is None:
        food, sport = "Periksa input", "Periksa input"
    elif bmi >= 30:
        food, sport = "Ganti nasi putih dengan nasi merah/ubi, kurangi gorengan.", "Jalan kaki 20–30 menit tiap hari."
    elif bmi >= 25:
        food, sport = "Kurangi minuman manis, perbanyak sayur & protein.", "Naik tangga dan stretching ringan."
    else:
        food, sport = "Pertahankan pola makan seimbang.", "Lanjutkan aktivitas rutin seperti yoga/jalan pagi."

    lifestyle = []
    if fcvc < 2: lifestyle.append("Tingkatkan porsi sayur.")
    if tue > 8: lifestyle.append("Kurangi waktu gadget.")
    if faf == 0: lifestyle.append("Mulai olahraga ringan.")
    lifestyle_txt = " ".join(lifestyle)

    bmi_txt = f"{bmi:.1f}" if bmi is not None else "—"
    return f"BMI {bmi_txt} ({cat}). {reco} Risiko diabetes: {score_txt}. Makanan: {food}. Aktivitas: {sport}. {lifestyle_txt}"

# Gauge BMI
def draw_gauge(bmi):
    if not PLOTLY_OK or bmi is None:
        st.metric("BMI", f"{bmi:.1f}" if bmi else "—")
        return
    if bmi < 18.5: color = "#33C4FF"
    elif bmi < 25: color = "#2ECC71"
    elif bmi < 30: color = "#F1C40F"
    else: color = "#E74C3C"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=bmi, title={"text": "BMI Score"},
                                 gauge={"axis": {"range": [10, 45]}, "bar": {"color": color}}))
    st.plotly_chart(fig, use_container_width=True)

# Pie chart nutrisi
def draw_macro(label):
    if not PLOTLY_OK:
        st.write("Target Nutrisi (Plotly tidak aktif)")
        return
    if "Obesity" in label:
        values, title = [20, 50, 30], "Low Carb"
    elif "Overweight" in label:
        values, title = [40, 40, 20], "Kurangi Lemak"
    else:
        values, title = [50, 30, 20], "Diet Normal"
    labels = ["Karbo", "Protein", "Lemak"]
    colors = ["#F4D03F", "#58D68D", "#5DADE2"]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker_colors=colors, sort=False)])
    fig.update_layout(title_text=f"Target Nutrisi ({title})")
    st.plotly_chart(fig, use_container_width=True)

# Dashboard
def dashboard():
    with st.sidebar:
        st.title(f"👤 Hi, {st.session_state['username']}")
        if st.button("Log Out"):
            st.session_state['logged_in'] = False
            st.experimental_rerun()
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

    try:
        bmi = weight / ((height / 100) ** 2)
    except:
        bmi = None

    st.title("🧬 AiLen Health Dashboard")
    col_left, col_right = st.columns([1, 1.3])
    with col_left:
        st.subheader("📊 Analisis Tubuh")
        st.info(f"Kategori BMI: {bmi:.1f}" if bmi else "Kategori BMI: —")
        draw_gauge(bmi)
        draw_macro("BMI")
    with col_right:
        st.subheader("💬 Chat dengan AiLen")
        chat_container = st.container()
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Tanya AiLen...")
            send_btn = st.form_submit_button("Kirim")
        if send_btn and user_input:
            st.session_state['chat_history'].append({"role": "user", "message": user_input})
            reply = get_bot_resp(height, weight, glucose, fcvc, tue, faf)
            st.session_state['chat_history'].append({"role": "bot", "message": reply})
            st.experimental_rerun()
        with chat_container:
            for chat in st.session_state['chat_history']:
                if chat["role"] == "user":
                    st.success(f"👤 Kamu: {chat['message']}")
                else:
                    st.info(f"🤖 AiLen: {chat['message']}")

# Main
if st.session_state['logged_in']:
    dashboard()
else:
    login_page()
