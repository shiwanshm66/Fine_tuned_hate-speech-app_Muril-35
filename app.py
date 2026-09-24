import streamlit as st
import torch
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================

MODEL_REPO = "ArpitSingh18/muril-hate-speech-arpit"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_NAME = "Fine-Tuned Hate Speech Detection Model"
AUTHOR_NAME = "Arpit Singh"
MODEL_VERSION = "v2.0 (MuRIL Fine-tuned)"

MODEL_PRECISION = 0.95
MODEL_RECALL = 0.94
MODEL_F1 = 0.945

# =========================
# PAGE SETUP
# =========================

st.set_page_config(page_title=PROJECT_NAME, page_icon="🧠", layout="centered")

# =========================
# CSS
# =========================

st.markdown("""
<style>
body { background-color: #0f172a; }
.main-title {
    text-align:center;font-size:2.8rem;font-weight:800;
    background:linear-gradient(90deg,#38bdf8,#22c55e);
    -webkit-background-clip:text;color:transparent;
}
.subtitle { text-align:center;color:#cbd5f5;font-size:1.1rem;margin-bottom:20px; }
.card {
    background:#020617;border-radius:18px;padding:25px;
    box-shadow:0 0 25px rgba(56,189,248,.15);margin-top:20px;
}
.result-box {border-radius:15px;padding:18px;margin-top:15px;font-size:1.3rem;font-weight:bold;text-align:center;}
.hate {background:linear-gradient(90deg,#7f1d1d,#ef4444);color:white;}
.clean {background:linear-gradient(90deg,#14532d,#22c55e);color:white;}
.metric-card {
    background:#020617;border-radius:14px;padding:15px;text-align:center;
    box-shadow:0 0 15px rgba(34,197,94,.15);
}
.footer {text-align:center;color:#94a3b8;margin-top:40px;font-size:.95rem;}
.badge {
    display:inline-block;padding:6px 12px;border-radius:12px;
    background:#1e293b;color:#38bdf8;font-size:0.8rem;margin:3px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CACHED)
# =========================

@st.cache_resource(show_spinner=True)
def load_model():
    st.write("🔤 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, use_fast=True)

    st.write("🧠 Loading model weights...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)

    model.to(DEVICE)
    model.eval()

    st.write("✅ Model loaded successfully!")
    return tokenizer, model


# IMPORTANT: actually load the model
tokenizer, model = load_model()

# =========================
# PREDICTION
# =========================

def predict(text):
    start = time.time()

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
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    latency = (time.time() - start) * 1000
    return probs, latency

# =========================
# HEADER
# =========================

st.markdown(f'<div class="main-title">{PROJECT_NAME}</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Fine-tuned MuRIL Transformer • Multilingual NLP (English, Hindi, Hinglish)</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center">
<span class="badge">Model: {MODEL_VERSION}</span>
<span class="badge">Device: {DEVICE.type.upper()}</span>
<span class="badge">Framework: PyTorch + Transformers</span>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUT
# =========================

st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("✍️ Enter text to classify:", height=130)

threshold = st.slider("Decision Threshold (Hate probability)", 0.1, 0.9, 0.5, 0.05)

c1, c2, c3 = st.columns(3)

with c1:
    if st.button("Example Hate"):
        user_input = "तुम बहुत बेकार इंसान हो और किसी काम के नहीं हो"

with c2:
    if st.button("Example Non-Hate"):
        user_input = "आज का दिन बहुत अच्छा था और मैंने बहुत कुछ सीखा"

with c3:
    classify_btn = st.button("🚀 Classify")

if classify_btn:
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        probs, latency = predict(user_input)

        hate_prob = probs[1]
        nonhate_prob = probs[0]

        pred = 1 if hate_prob >= threshold else 0
        confidence = max(hate_prob, nonhate_prob)

        if pred == 1:
            st.markdown('<div class="result-box hate">🚨 HATE SPEECH DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box clean">✅ NON-HATE SPEECH</div>', unsafe_allow_html=True)

        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence*100:.2f}%**")
        st.write(f"Inference time: **{latency:.1f} ms**")

        st.markdown("#### 📊 Prediction Probabilities")
        chart_data = pd.DataFrame({
            "Class": ["Non-Hate", "Hate"],
            "Probability": [nonhate_prob, hate_prob]
        })
        st.bar_chart(chart_data.set_index("Class"))

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# BATCH CSV
# =========================

st.markdown("### 📁 Batch Prediction (CSV Upload)")
uploaded = st.file_uploader("Upload CSV with column name: text", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    results = []

    for t in df["text"].astype(str).tolist():
        probs, _ = predict(t)
        results.append("HATE" if probs[1] >= threshold else "NON-HATE")

    df["prediction"] = results
    st.dataframe(df)
    st.download_button("Download results", df.to_csv(index=False), "predictions.csv")

# =========================
# METRICS
# =========================

st.markdown("### 📊 Model Performance")

m1, m2, m3 = st.columns(3)
for col, name, val in zip([m1, m2, m3],
                           ["Precision", "Recall", "F1-Score"],
                           [MODEL_PRECISION, MODEL_RECALL, MODEL_F1]):
    with col:
        st.markdown(f"""
        <div class="metric-card">
        <h3>{name}</h3>
        <h2>{val:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

# =========================
# DESCRIPTION
# =========================

with st.expander("📌 Project Description"):
    st.markdown(f"""
**{PROJECT_NAME}** is a production-ready multilingual NLP system for classifying social media text into **Hate** and **Non-Hate**.

### Highlights
- Fine-tuned Google MuRIL Transformer
- English, Hindi & Hinglish support
- Weighted loss for class imbalance
- UTF-8 repaired dataset pipeline
- HuggingFace Hub model hosting
- Streamlit Cloud deployment
- Batch & real-time inference

### Use Cases
- Content moderation
- Toxic comment filtering
- Online safety platforms
- NLP research & AI portfolios

Developed by **{AUTHOR_NAME}** as an end-to-end AI engineering project.
""")

# =========================
# FOOTER
# =========================

st.markdown(f'<div class="footer">Developed by <b>{AUTHOR_NAME}</b> | Fine-Tuned Multilingual Hate Speech Detection System</div>', unsafe_allow_html=True)
