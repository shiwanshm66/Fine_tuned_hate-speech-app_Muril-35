import pandas as pd
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import chardet

# =========================
# CONFIG
# =========================
RAW_TEST_FILE = Path("/home/lambd/ml-projects/hate_speech_prediction/data/raw/SubTask-B-test.csv")
OUTPUT_FILE = Path("data/predictions/test_predictions.csv")
MODEL_PATH = "models/muril_hate_model_v2"
THRESHOLD = 0.50   # change to 0.17 if you want high recall
BATCH_SIZE = 16
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# DETECT ENCODING
# =========================
def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw = f.read(50000)
    return chardet.detect(raw)["encoding"]

# =========================
# LOAD & FIX TEST DATA
# =========================
def load_and_fix_test_csv():
    print("Detecting encoding...")
    encoding = detect_encoding(RAW_TEST_FILE)
    print("Detected encoding:", encoding)

    print("Loading test.csv...")
    df = pd.read_csv(RAW_TEST_FILE, encoding=encoding)

    # Fix column name
    if "tweet" in df.columns:
        df.rename(columns={"tweet": "text"}, inplace=True)
    elif "text" not in df.columns:
        df.columns = ["text"]

    # Fix datatype
    df["text"] = df["text"].astype(str)

    # Drop empty rows
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""]

    # Clean text
    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("Total samples:", len(df))
    return df

# =========================
# LOAD MODEL
# =========================
def load_model():
    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# =========================
# BATCH INFERENCE
# =========================
def predict_batch(texts, tokenizer, model):
    predictions = []
    probabilities = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        hate_probs = probs[:, 1].cpu().numpy()

        for p in hate_probs:
            label = "Hate" if p >= THRESHOLD else "Non-Hate"
            predictions.append(label)
            probabilities.append(float(p))

    return predictions, probabilities

# =========================
# MAIN
# =========================
def main():
    print("\n========== TEST CSV PREDICTION PIPELINE ==========\n")

    df = load_and_fix_test_csv()
    tokenizer, model = load_model()

    print("Running predictions...")
    preds, probs = predict_batch(df["clean_text"].tolist(), tokenizer, model)

    df["prediction"] = preds
    df["hate_probability"] = probs

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("\n==============================")
    print("PREDICTION COMPLETED ✅")
    print("Saved to:", OUTPUT_FILE.resolve())
    print("==============================\n")

    print(df[["text", "prediction", "hate_probability"]].head())

if __name__ == "__main__":
    main()
