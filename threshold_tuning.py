import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "models/muril_hate_model_v2"
DATA_PATH = "data/tokenized/val"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================
# LOAD DATA
# ==============================
print("Loading validation dataset...")
dataset = load_from_disk(DATA_PATH)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch]),
    }


loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ==============================
# LOAD MODEL
# ==============================
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ==============================
# COLLECT PROBABILITIES
# ==============================
all_labels = []
all_probs = []

print("Running inference...")

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # hate prob

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# ==============================
# THRESHOLD SEARCH
# ==============================
print("\nSearching best threshold for HATE class...\n")

results = []

for threshold in np.arange(0.05, 0.96, 0.01):
    preds = (all_probs >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average=None, zero_division=0
    )

    hate_precision = precision[1]
    hate_recall = recall[1]
    hate_f1 = f1[1]

    results.append((threshold, hate_precision, hate_recall, hate_f1))

# Sort by F1 descending
results.sort(key=lambda x: x[3], reverse=True)

# ==============================
# DISPLAY RESULTS
# ==============================
best = results[0]

print("=======================================")
print(" BEST THRESHOLD FOUND ")
print("=======================================")
print(f"Threshold  : {best[0]:.2f}")
print(f"Precision  : {best[1]:.4f}")
print(f"Recall     : {best[2]:.4f}")
print(f"F1-score   : {best[3]:.4f}")
print("=======================================\n")

print("Top 10 thresholds:")

for r in results[:10]:
    print(f"Threshold={r[0]:.2f} | Precision={r[1]:.3f} | Recall={r[2]:.3f} | F1={r[3]:.3f}")

print("\nThreshold tuning completed ✅")
