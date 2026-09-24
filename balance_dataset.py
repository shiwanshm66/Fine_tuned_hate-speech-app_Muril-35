import pandas as pd
from pathlib import Path
from sklearn.utils import resample

# ============================
# Paths
# ============================
BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

INPUT_FILE = BASE_DIR / "data/processed/train_preprocessed.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/train_balanced.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\n==============================")
print("STEP 3: DATASET BALANCING")
print("==============================")

# ============================
# Load dataset (force UTF-8)
# ============================
df = pd.read_csv(INPUT_FILE, encoding="utf-8")

print("\nOriginal shape:", df.shape)
print("\nOriginal class distribution:")
print(df["label"].value_counts())

# ============================
# Split classes
# ============================
df_majority = df[df["label"] == 0]
df_minority = df[df["label"] == 1]

print("\nMajority (non-hate):", len(df_majority))
print("Minority (hate):", len(df_minority))

# ============================
# Undersample majority (1:2 ratio)
# ============================
target_majority_size = len(df_minority) * 2

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=target_majority_size,
    random_state=42
)

# ============================
# Combine & shuffle
# ============================
df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================
# Save
# ============================
df_balanced.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ============================
# Report
# ============================
print("\nBalanced shape:", df_balanced.shape)
print("\nBalanced class distribution:")
print(df_balanced["label"].value_counts())

print("\nSample NON-HATE text:")
print(df_balanced[df_balanced["label"] == 0]["text"].iloc[0])

print("\nSample HATE text:")
print(df_balanced[df_balanced["label"] == 1]["text"].iloc[0])

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nBalancing completed successfully ✅")
print("==============================\n")
