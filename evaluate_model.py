import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ============================
# Paths
# ============================
BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

MODEL_DIR = BASE_DIR / "models/muril_hate_model"
VAL_DATA_DIR = BASE_DIR / "data/tokenized/val"

# ============================
# Load model
# ============================
print("\n==============================")
print("STEP 6: MODEL EVALUATION")
print("==============================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ============================
# Load validation dataset
# ============================
val_dataset = load_from_disk(VAL_DATA_DIR)

print("Validation samples:", len(val_dataset))

# ============================
# Prediction loop
# ============================
y_true = []
y_pred = []

print("\nRunning inference on validation set...")

with torch.no_grad():
    for sample in val_dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        label = int(sample["label"])

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        pred = torch.argmax(logits, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

# ============================
# Metrics
# ============================
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
cm = confusion_matrix(y_true, y_pred)

# ============================
# Report
# ============================
print("\n==============================")
print("EVALUATION RESULTS")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Non-Hate", "Hate"]))

print("\nEvaluation completed successfully ✅")
print("==============================\n")
