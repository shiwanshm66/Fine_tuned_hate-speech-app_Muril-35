import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

sns.set(style="whitegrid")

# =========================
# Configuration
# =========================
DATA_PATH = Path("/home/lambd/ml-projects/hate_speech_prediction/data/raw/train_utf8.csv")
TEXT_COL = "tweet"
LABEL_COL = "label"
ENCODING = "utf-8"

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH, encoding=ENCODING)

print("\n==============================")
print("DATASET LOADED SUCCESSFULLY")
print("==============================")

# =========================
# Shape
# =========================
print("\nDataset Shape (rows, columns):", df.shape)

# =========================
# Column info
# =========================
print("\nColumn Information:")
print(df.info())

# =========================
# Rename column for consistency
# =========================
df.rename(columns={TEXT_COL: "text"}, inplace=True)

# =========================
# Missing values
# =========================
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# =========================
# Duplicate rows
# =========================
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")

# =========================
# Class distribution
# =========================
class_counts = df["label"].value_counts().sort_index()
class_percent = df["label"].value_counts(normalize=True).sort_index() * 100

print("\nClass Distribution (Counts):")
print(class_counts)

print("\nClass Distribution (Percentage):")
print(class_percent.round(2))

# Plot class distribution
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Class Distribution (0 = Non-Hate, 1 = Hate)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# Text length analysis
# =========================
df["char_length"] = df["text"].astype(str).apply(len)
df["word_length"] = df["text"].astype(str).apply(lambda x: len(str(x).split()))

print("\nText Length (Characters) Statistics:")
print(df["char_length"].describe())

print("\nText Length (Words) Statistics:")
print(df["word_length"].describe())

# Histogram: character length
plt.figure(figsize=(8,4))
sns.histplot(df["char_length"], bins=50)
plt.title("Character Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Histogram: word length
plt.figure(figsize=(8,4))
sns.histplot(df["word_length"], bins=50)
plt.title("Word Length Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Boxplot by class
plt.figure(figsize=(8,4))
sns.boxplot(x="label", y="char_length", data=df)
plt.title("Character Length by Class")
plt.xlabel("Class")
plt.ylabel("Characters")
plt.tight_layout()
plt.show()

# =========================
# Most common words
# =========================
def get_top_words(text_series, n=20):
    all_words = []
    for text in text_series:
        words = str(text).split()
        all_words.extend(words)
    return Counter(all_words).most_common(n)

top_non_hate = get_top_words(df[df["label"] == 0]["text"])
top_hate = get_top_words(df[df["label"] == 1]["text"])

print("\nTop 20 words in NON-HATE class:")
print(top_non_hate)

print("\nTop 20 words in HATE class:")
print(top_hate)

# =========================
# Minority class ratio
# =========================
minority_ratio = class_counts.min() / class_counts.max()

print(f"\nMinority / Majority ratio: {minority_ratio:.3f}")

if minority_ratio < 0.4:
    print("⚠️ Severe class imbalance detected → Class weighting REQUIRED.")
elif minority_ratio < 0.7:
    print("⚠️ Moderate class imbalance → Class weighting recommended.")
else:
    print("✅ Class distribution looks balanced.")

# =========================
# Sample texts
# =========================
print("\nSample NON-HATE text:")
print(df[df["label"] == 0]["text"].iloc[0])

print("\nSample HATE text:")
print(df[df["label"] == 1]["text"].iloc[0])

# =========================
# Final EDA summary
# =========================
print("\n==============================")
print("EDA SUMMARY")
print("==============================")
print(f"Total samples: {len(df)}")
print(f"Non-hate samples: {class_counts.get(0,0)}")
print(f"Hate samples: {class_counts.get(1,0)}")
print(f"Duplicate rows: {duplicate_count}")
print(f"Average characters: {df['char_length'].mean():.1f}")
print(f"Average words: {df['word_length'].mean():.1f}")
print("==============================")
print("EDA COMPLETED SUCCESSFULLY")


