import os
import re

import pandas as pd

INPUT_FILE = "data/stress-minimal/single.csv"  # Name of your source file
OUTPUT_FILE = "output_split_single.csv"  # Name of the file to be created


def split_sentence_phoneme_pairs(row):
    if pd.isna(row["sentence"]) or pd.isna(row["phoneme"]):
        return []

    sentences = re.split(r"(?<=[.!?]) +", str(row["sentence"]))

    phonemes = str(row["phoneme"]).split()

    results = []
    current_phoneme_idx = 0

    for sent in sentences:
        word_count = len(re.findall(r"\w+", sent))

        sent_phonemes = phonemes[current_phoneme_idx : current_phoneme_idx + word_count]

        results.append(
            {
                "index": row["index"],
                "sentence": sent.strip(),
                "phoneme": " ".join(sent_phonemes),
            }
        )

        current_phoneme_idx += word_count

    return results


if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    return

print(f"Reading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

new_rows = []
for _, row in df.iterrows():
    new_rows.extend(split_sentence_phoneme_pairs(row))

output_df = pd.DataFrame(new_rows)

output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"Successfully processed {len(df)} rows into {len(output_df)} sentences.")
print(f"Saved results to {OUTPUT_FILE}")
