import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

# ============================
# Config
# ============================
BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

INPUT_FILE = BASE_DIR / "data/processed/train_balanced.csv"
OUTPUT_DIR = BASE_DIR / "data/tokenized"

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Load dataset
# ============================
print("\n==============================")
print("STEP 4: TOKENIZATION PIPELINE")
print("==============================")

df = pd.read_csv(INPUT_FILE, encoding="utf-8")

print("\nLoaded dataset shape:", df.shape)
print("Class distribution:")
print(df["label"].value_counts())

# ============================
# Train / validation split
# ============================
train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

print("\nTrain shape:", train_df.shape)
print("Validation shape:", val_df.shape)

# ============================
# Load tokenizer
# ============================
print("\nLoading MuRIL tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================
# Convert to HF Dataset
# ============================
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# ============================
# Tokenization function
# ============================
def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# ============================
# Tokenize
# ============================
print("\nTokenizing training data...")
train_dataset = train_dataset.map(tokenize_function, batched=True)

print("Tokenizing validation data...")
val_dataset = val_dataset.map(tokenize_function, batched=True)

# ============================
# Keep required columns
# ============================
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# ============================
# Save to disk
# ============================
train_dataset.save_to_disk(OUTPUT_DIR / "train")
val_dataset.save_to_disk(OUTPUT_DIR / "val")

print("\nTokenized datasets saved to:")
print(OUTPUT_DIR)

print("Sample tokenized record saved successfully.")


print("\nTokenization completed successfully ✅")
print("==============================\n")
