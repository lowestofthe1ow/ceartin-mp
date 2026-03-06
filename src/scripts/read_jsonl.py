import json
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from src.utils.homographs import fill_template, homographs

FILE_PATH = "results_gemini_3.jsonl"
OUTPUT_CSV = "phonetic_tatoeba_gemini_3.csv"

PHONEME_INVENTORY = [
    "'",
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "ŋ",
    "ɕ",
    "ə",
    "ɡ",
    "ɹ",
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",
    "ˌ",
    # NOTE: We allow both normal and joined tie bars
    "\u0361",
    " ‍͡ ",
    " ",
]


def normalize_characters(text):
    """Normalizes the characters in a string to the phoneme inventory above"""

    if not text:
        return ""

    # TODO: Can look into the following mappings:
    # "ɛ": "e", "ɪ": "i", "ʊ": "u", "ɔ": "o", "ɑ": "a",
    replacements = {
        "g": "ɡ",
        "r": "ɾ",
        ",": "ˌ",
        "Ɂ": "ʔ",
        ".": "",
        "‍": "",  # Zero-width joiner that Gemini hallucinates sometimes
    }

    text = text.replace(".", "")  # Remove syllable markers

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


def validate_characters(answers):
    """Checks if the string contains characters not in the phoneme inventory"""

    for word in answers:
        for char in word:
            if char not in PHONEME_INVENTORY:
                try:
                    char_name = unicodedata.name(char)
                except ValueError:
                    char_name = "Unknown"

                # Print exactly what caused the error
                print(f"Failed to process word: '{word}'")
                print(f"Error at '{char}' (U+{ord(char):04X}: {char_name})")

                return False

    return True


# Load Tatoeba dataset
print("Loading Tatoeba dataset...")
dataset = load_dataset("tatoeba", "en-tl", lang1="en", lang2="tl")
tatoeba_sentences = [item["tl"] for item in dataset["train"]["translation"]]

results_list = []
error_count = 0

with open(FILE_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Processing JSONL", unit="line"):

        # TODO: This part needs cleaning up

        data = json.loads(line)
        idx = data.get("index")

        if idx is None or (idx - 1) >= len(tatoeba_sentences) or (idx - 1) < 0:
            continue

        sentence = tatoeba_sentences[idx - 1]

        if not sentence or not str(sentence).strip():
            continue

        try:
            _, _, output_template = homographs(sentence)
        except Exception:
            continue

        output_string = None
        status = None

        if data.get("success") is True:
            answers = data["content"]["answers"]
            answers = [normalize_characters(a) for a in answers]

            if validate_characters(answers):
                try:
                    output_string = fill_template(output_template, answers)
                    status = "success"
                except StopIteration:
                    error_count += 1
                    continue
                except Exception:
                    error_count += 1
                    continue
        elif data.get("error") == "ERR_EMPTY_PROMPT":
            # Case when no ambiguous words were found
            output_string = " ".join(output_template)
            status = "empty_prompt_fallback"

        if output_string and str(output_string).strip():
            results_list.append(
                {
                    "index": idx,
                    "original_sentence": sentence,
                    "processed_output": output_string,
                    "status": status,
                }
            )

df = pd.DataFrame(results_list)
df = df.drop(columns=["status"])
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("\n" + "=" * 40)
print("Summary")
print("-" * 40)
print(f"Total lines in JSONL: {total_lines}")
print(f"Valid entries: {len(df)}")
print(f"Template mismatches : {error_count}")
print("=" * 40)
