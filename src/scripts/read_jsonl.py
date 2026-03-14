"""
This script parses the .jsonl output of the scripts/query_gemini_tatoeba.py to
produce a cleaner CSV file for use in training
"""

import argparse
import json
import re
import unicodedata

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from src.utils.homographs import fill_template, homographs

# TODO: Use argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    default="data/phonetic_newsph-nli/results_gemini_2.5_flash_lite.jsonl",
)
parser.add_argument("--output", type=str, default="output_from_jsonl.csv")

args = parser.parse_args()


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
    # We allow both normal and joined tie bars
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
        # For Gemini 2.5-Flash-Lite output
        "ɛ": "e",
        "ɪ": "i",
        "ʊ": "u",
        "ɔ": "o",
        "ɑ": "a",
        "æ": "a",
        ":": "",
        "꞉": "",
        "ː": "",
        "\u200b": "",  # Zero-width space
        "ɐ": "a",
        "á": "a",
        "ɭ": "l",
        "ʤ": "dʒ",
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
print("Loading dataset...")

with open("data/newsph-nli/newsph-nli.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

results_list = []
error_count = 0
invalid_responses = 0

with open(args.path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(args.path, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Processing JSONL", unit="line"):

        # TODO: This part needs cleaning up

        data = json.loads(line)
        idx = data.get("index")

        if idx is None or (idx - 1) >= len(sentences) or (idx - 1) < 0:
            continue

        sentence = sentences[idx - 1]

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
        else:
            invalid_responses += 1
            continue

        if output_string and str(output_string).strip():
            results_list.append(
                {
                    "index": idx,
                    "sentence": sentence,
                    "phoneme": output_string,
                    "status": status,
                }
            )

df = pd.DataFrame(results_list)
df = df.drop(columns=["status"])
df.to_csv(args.output, index=False, encoding="utf-8")

print("=" * 40)
print("Summary")
print("-" * 40)
print(f"Total lines in JSONL: {total_lines}")
print(f"Valid entries: {len(df)}")
print(f"Template mismatches : {error_count}")
print(f"Invalid API responses: {invalid_responses}")
print("=" * 40)
