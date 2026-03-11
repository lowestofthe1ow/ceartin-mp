"""
This script is used to write Tatoeba corpus data into a .txt file.
"""

import os
from pathlib import Path

from datasets import load_dataset

OUTPUT_PATH = Path("data/tatoeba")
OUTPUT_FILENAME = "tatoeba.txt"

print("Loading Tatoeba dataset...")
dataset = load_dataset("tatoeba", "en-tl", lang1="en", lang2="tl")
sentences = [item["tl"] for item in dataset["train"]["translation"]]

os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(OUTPUT_PATH / OUTPUT_FILENAME, "w") as file:
    for item in sentences:
        file.write(f"{item}\n")

print(f"Wrote {len(sentences)} lines into {OUTPUT_PATH / OUTPUT_FILENAME}")
