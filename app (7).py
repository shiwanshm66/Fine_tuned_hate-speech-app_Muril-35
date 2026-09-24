import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# ============================
# Config
# ============================

BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

MODEL_DIR = BASE_DIR / "models/muril_hate_model_v2"
MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Page setup
# ============================

st.set_page_config(
    page_title="Hate Speech AI Detector",
    page_icon="🧠",
    layout="centered"
)

# ============================
# Custom CSS
# ============================

st.markdown("""
<style>
body {
    background: #0f172a;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    font-size: 1.1rem;
}

.card {
    background: #020617;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0px 0px 25px rgba(56,189,248,0.15);
    margin-top: 20px;
}

.result-box {
    border-radius: 15px;
    padding: 20px;
    margin-top: 15px;
    font-size: 1.3rem;
    font-weight: bold;
    text-align: center;
}

.hate {
    background: linear-gradient(90deg, #7f1d1d, #ef4444);
    color: white;
}

.clean {
    background: linear-gradient(90deg, #14532d, #22c55e);
    color: white;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ============================
# Load model
# ============================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ============================
# Prediction
# ============================

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred].item()
    return pred, confidence

# ============================
# UI Layout
# ============================

st.markdown('<div class="main-title">Fine-Tuned Hate Speech Detection Model</div>', unsafe_allow_html=True)


st.markdown('<div class="subtitle">Multilingual Transformer (MuRIL) • Real-time NLP Classification</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter text (English / Hindi / Hinglish)", height=140)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Example: Hate"):
        user_input = "तुम बहुत घटिया इंसान हो"

with col2:
    if st.button("Example: Non-Hate"):
        user_input = "Have a wonderful day my friend"

with col3:
    classify = st.button("🚀 Classify")

if classify:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("🧠 Analyzing with Transformer model..."):
            pred, conf = predict(user_input)

        if pred == 1:
            st.markdown(f'<div class="result-box hate">🚨 HATE SPEECH DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-box clean">✅ NON-HATE SPEECH</div>', unsafe_allow_html=True)

        st.progress(int(conf * 100))
        st.write(f"Confidence: **{conf*100:.2f}%**")

st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Info section
# ============================

with st.expander("📌 Model & Project Details"):
    st.markdown("""
    **Model:** Google MuRIL (Multilingual BERT)  
    **Classes:** Hate / Non-Hate  
    **Languages:** English, Hindi, Hinglish  
    **Training:** Weighted loss for class imbalance  
    **Accuracy:** >95%  
    **Dataset:** Cleaned & UTF-8 fixed social media text  
    **Frameworks:** PyTorch, HuggingFace, Streamlit  
    """)

st.markdown('<div class="footer">Built by Arpit Singh • AI/ML Engineer Portfolio Project</div>', unsafe_allow_html=True)
