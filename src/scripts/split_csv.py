import argparse
import os
import re

import pandas as pd

parser = argparse.ArgumentParser(
    description="Split CSV rows into individual sentence-phoneme pairs."
)
parser.add_argument("input")
parser.add_argument("output")
args = parser.parse_args()

INPUT_FILE = args.input
OUTPUT_FILE = args.output


def split_sentence_phoneme_pairs(row):
    if pd.isna(row["sentence"]) or pd.isna(row["phoneme"]):
        return []

    sentences = re.split(r"(?<=[.!?]) +", str(row["sentence"]))
    phonemes = str(row["phoneme"]).split()

    results = []
    current_phoneme_idx = 0

    for sent in sentences:
        words_in_sent = sent.strip().split()
        word_count = len(words_in_sent)

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


# Set up argument parsing
if not os.path.exists(INPUT_FILE):
    print(f"Error: {INPUT_FILE} not found.")
    exit(1)

print(f"Reading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)

new_rows = []
for _, row in df.iterrows():
    new_rows.extend(split_sentence_phoneme_pairs(row))

output_df = pd.DataFrame(new_rows)

output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"Successfully processed {len(df)} rows into {len(output_df)} sentences.")
print(f"Saved results to {OUTPUT_FILE}")
