import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/lambd/ml-projects/hate_speech_prediction")

RAW_FILE = BASE_DIR / "data/raw/SubTask-B-train.csv"
OUTPUT_FILE = BASE_DIR / "data/processed/train_master_clean.csv"

def try_fix(text):
    """
    Fix UTF-8 wrongly decoded as latin1.
    """
    if not isinstance(text, str):
        return "", False

    try:
        fixed = text.encode("latin1").decode("utf-8")
    except:
        fixed = text

    # count Hindi chars
    def hindi_ratio(s):
        if len(s) == 0:
            return 0
        return sum(0x0900 <= ord(c) <= 0x097F for c in s) / len(s)

    if hindi_ratio(fixed) > hindi_ratio(text):
        return fixed, True
    else:
        return text, False


def is_garbage(text):
    bad = ["à¤", "Ã", "â€", "¤", "�"]
    return sum(text.count(b) for b in bad) > 3


print("Loading RAW file using latin1 (byte-safe)...")
df = pd.read_csv(RAW_FILE, encoding="latin1")

df.rename(columns={"tweet": "text"}, inplace=True)

fixed_count = 0
garbage_count = 0
clean_texts = []

for t in df["text"]:
    fixed, changed = try_fix(t)
    if changed:
        fixed_count += 1

    if is_garbage(fixed):
        clean_texts.append(None)
        garbage_count += 1
    else:
        clean_texts.append(fixed)

df["text"] = clean_texts

before = len(df)
df = df.dropna(subset=["text"])
after = len(df)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("\n========== REBUILD REPORT ==========")
print("Rows original :", before)
print("Rows recovered:", after)
print("Rows removed  :", before - after)
print("Rows fixed    :", fixed_count)
print("Garbage rows  :", garbage_count)
print("Saved to      :", OUTPUT_FILE)
print("===================================")
print("\nRebuilding completed successfully ✅")