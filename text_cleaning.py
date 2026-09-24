import pandas as pd
import re
from pathlib import Path

# ============================
# Paths
# ============================
BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

INPUT_FILE = BASE_DIR / "data/processed/train_master_clean.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/train_preprocessed.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================
# Cleaning function
# ============================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove user mentions
    text = re.sub(r"@\w+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ============================
# Load data
# ============================
print("\n==============================")
print("STEP 2: PREPROCESSING DATASET")
print("==============================")

df = pd.read_csv(INPUT_FILE, encoding="utf-8")

print("\nOriginal shape:", df.shape)
print("\nOriginal class distribution:")
print(df["label"].value_counts())

# ============================
# Remove duplicates
# ============================
before = len(df)
df = df.drop_duplicates(subset=["text", "label"])
after = len(df)

print(f"\nDuplicates removed: {before - after}")

# ============================
# Clean text
# ============================
print("\nCleaning text...")
df["text"] = df["text"].apply(clean_text)

# ============================
# Remove empty rows
# ============================
before_empty = len(df)
df = df[df["text"].str.len() > 0]
after_empty = len(df)

print(f"Empty rows removed: {before_empty - after_empty}")

# ============================
# Save
# ============================
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ============================
# Final report
# ============================
print("\n==============================")
print("PREPROCESSING REPORT")
print("==============================")

print("Final shape:", df.shape)
print("\nFinal class distribution:")
print(df["label"].value_counts())

print("\nSample NON-HATE:")
print(df[df["label"] == 0]["text"].iloc[0])

print("\nSample HATE:")
print(df[df["label"] == 1]["text"].iloc[0])

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nPreprocessing completed successfully ✅")
print("==============================\n")
