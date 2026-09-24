import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# ============================
# Config
# ============================
BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

MODEL_NAME = "google/muril-base-cased"
TRAIN_DATA_DIR = BASE_DIR / "data/tokenized/train"
VAL_DATA_DIR = BASE_DIR / "data/tokenized/val"
OUTPUT_MODEL_DIR = BASE_DIR / "models/muril_hate_model_v2"

NUM_LABELS = 2
EPOCHS = 7
BATCH_SIZE = 8
LR = 1.5e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# ============================
# Load datasets
# ============================
print("\n==============================")
print("STEP 5 (MODERN): MODEL TRAINING")
print("==============================")

train_dataset = load_from_disk(TRAIN_DATA_DIR)
val_dataset = load_from_disk(VAL_DATA_DIR)

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))

# ============================
# Compute class weights
# ============================
labels = [int(x) for x in train_dataset["label"]]
counts = Counter(labels)

num_neg = counts.get(0, 0)
num_pos = counts.get(1, 0)

print("\nClass counts:", counts)

if num_neg == 0 or num_pos == 0:
    raise ValueError("Training split contains only one class!")

weight_for_0 = (num_neg + num_pos) / (2.0 * num_neg)
weight_for_1 = (num_neg + num_pos) / (2.0 * num_pos)

class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float)

print("Class weights:", class_weights.tolist())

# ============================
# Load model
# ============================
print("\nLoading MuRIL model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# ============================
# Custom Trainer (weighted loss)
# ============================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ============================
# Metrics
# ============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# ============================
# Training arguments (modern)
# ============================
training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "training_output_v2"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    logging_dir=str(BASE_DIR / "logs"),
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# ============================
# Trainer
# ============================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ============================
# Train
# ============================
print("\nStarting training...\n")
trainer.train()

# ============================
# Save model
# ============================
OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(OUTPUT_MODEL_DIR))

print("\nModel saved to:", OUTPUT_MODEL_DIR)
print("\nTraining completed successfully ✅")
print("==============================\n")
