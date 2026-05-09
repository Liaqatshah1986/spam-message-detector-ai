import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="📩",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }

    h1 {
        color: #38bdf8;
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        background-color: #38bdf8;
        color: black;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About")
st.sidebar.info(
    """
    AI-powered spam message detector.

    Built using:
    - Python
    - Scikit-learn
    - Streamlit
    """
)

# ---------------- MAIN UI ----------------
st.title("📩 Spam Message Detector")

message = st.text_area("✉ Enter Message")

if st.button("Check Message"):

    if message.strip() == "":
        st.warning("⚠ Please enter a message.")

    else:
        msg_vec = vectorizer.transform([message])

        prediction = model.predict(msg_vec)[0]

        probability = model.predict_proba(msg_vec)[0]

        spam_confidence = probability[1] * 100
        ham_confidence = probability[0] * 100

        st.subheader("Result")

        if prediction == 1:
            st.error("🚨 Spam Message Detected")
            st.write(f"Spam Confidence: {spam_confidence:.2f}%")

        else:
            st.success("✅ Not Spam")
            st.write(f"Safe Message Confidence: {ham_confidence:.2f}%")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Machine Learning")