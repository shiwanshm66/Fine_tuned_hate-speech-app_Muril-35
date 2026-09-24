import torch
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/muril_hate_model_v2"
THRESHOLD = 0.17   # change to 0.17 if you want higher recall
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD MODEL & TOKENIZER
# ==============================
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

# ==============================
# CLEANING FUNCTION
# ==============================
def clean_text(text: str) -> str:
    text = text.strip()
    return text

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict(text: str):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    prob_non_hate = probs[0].item()
    prob_hate = probs[1].item()

    label = "Hate" if prob_hate >= THRESHOLD else "Non-Hate"

    return label, prob_hate, prob_non_hate

# ==============================
# CLI MODE
# ==============================
if __name__ == "__main__":
    print("\nMuRIL Hate Speech Predictor")
    print("Type 'exit' to quit\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() == "exit":
            break

        label, p_hate, p_non = predict(text)

        print("\nPrediction:", label)
        print(f"Hate probability     : {p_hate:.4f}")
        print(f"Non-hate probability : {p_non:.4f}")
        print("-" * 40)
