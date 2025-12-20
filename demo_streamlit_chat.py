import streamlit as st
import requests

st.title("Diabetes Risk Chatbot")

st.write("Contoh input: `tinggi 160 berat 70 glukosa 110`")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ketik pesan ke chatbot:")

if st.button("Kirim") and user_input.strip():
    try:
        r = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"message": user_input},
            timeout=5,
        )
        data = r.json()
        bot_reply = data.get("response", str(data))
    except Exception as e:
        bot_reply = f"Request failed: {e}"

    st.session_state.history.append(("Kamu", user_input))
    st.session_state.history.append(("Bot", bot_reply))

# Tampilkan history chat
for sender, msg in st.session_state.history:
    if sender == "Kamu":
        st.markdown(f"**{sender}:** {msg}")
    else:
        st.markdown(f"**{sender}:** {msg}")
