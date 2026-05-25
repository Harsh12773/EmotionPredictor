import streamlit as st
import joblib
import re

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #f8fbff, #edf4ff);
}

.main {
    background-color: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #2563eb;
    font-size: 42px !important;
}

.subtitle {
    text-align: center;
    color: #64748b;
    margin-bottom: 1.5rem;
    font-size: 17px;
}

textarea {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 16px !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3.1em;
    font-size: 17px;
    font-weight: 600;
}

.stButton > button:hover {
    opacity: 0.95;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CLEANING FUNCTION ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ---------------- EMOTION LABELS ----------------
emotion_labels = {
    0: "Sadness",
    1: "Anger",
    2: "Love",
    3: "Surprise",
    4: "Fear",
    5: "Joy"
}

# ---------------- UI ----------------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("EmotionSense AI")

st.markdown(
    "<div class='subtitle'>Emotion detection from text using NLP and Machine Learning</div>",
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Enter your text",
    placeholder="Example: I feel really happy and motivated today.",
    height=180
)

if st.button("Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned_text = clean_text(user_input)

        vector_input = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector_input)[0]

        emotion = emotion_labels[prediction]

        st.success(f"Predicted Emotion: {emotion}")

st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.caption("Built using Streamlit, Scikit-learn and NLP")