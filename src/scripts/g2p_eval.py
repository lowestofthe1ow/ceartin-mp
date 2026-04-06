"""
This script evaluates a model checkpoint on a given test set by calculating PER
and PFER.
"""

# TODO: This needs work.

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import panphon
import panphon.distance
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from datasets import concatenate_datasets
from src.utils.dataset_from_csv import dataset_from_csv, dataset_from_csv_list


def normalize_characters(text):
    """Normalizes the characters in a string to the phoneme inventory above"""

    if not text:
        return ""

    # TODO: Can look into the following mappings:
    # "ɛ": "e", "ɪ": "i", "ʊ": "u", "ɔ": "o", "ɑ": "a",
    replacements = {
        "g": "ɡ",
        "r": "ɾ",
        "ɹ": "ɾ",
        ",": "ˌ",
        "Ɂ": "ʔ",
        ".": "",
        "ˈ": "'",
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
        "ʌ": "a",
        "ɭ": "l",
        "ʤ": "dʒ",
        "ɕ": "ʃ",
    }

    text = text.replace(".", "")  # Remove syllable markers

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(r"\s+", " ", text).strip()


DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = "data/combined.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--checkpoint-path", default="")
parser.add_argument(
    "--dataset",
    default="tatoeba",
    choices=["tatoeba", "newsph-nli", "combined", "manual"],
)
parser.add_argument("--base-model", action="store_true")
parser.add_argument("--epitran", action="store_true")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
# Use default tokenizer with CharsiuG2P ByT5
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

# Set up dataset CSV paths
if args.dataset == "tatoeba":
    dataset = ["data/tatoeba/phonetic_tatoeba_gemini_3.csv"]
    split_dataset = dataset_from_csv_list(dataset, tokenizer)
elif args.dataset == "newsph-nli":
    dataset = ["data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv"]
    split_dataset = dataset_from_csv_list(dataset, tokenizer)
elif args.dataset == "combined":
    dataset = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    ]
    split_dataset = dataset_from_csv_list(dataset, tokenizer)
elif args.dataset == "manual":
    dataset = "data/manual_set.csv"

    # TODO: Dumb hack but whatever lol
    split_dataset = dataset_from_csv(dataset, tokenizer)
    split_dataset["test"] = concatenate_datasets(list(split_dataset.values()))

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.epitran:
    epi = epitran.Epitran("tgl-Latn")
else:
    load_path = args.model_id if args.base_model else args.checkpoint_path
    model = T5ForConditionalGeneration.from_pretrained(load_path)
    model.to(device)
    model.eval()

# Use Panphon's built in tools, less of a headache this way
ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

# Get the test split
test_set = split_dataset["test"]

# Running sums
total_per_dist = 0
total_pfer_dist = 0
total_cer_dist = 0
total_phonemes = 0
total_chars = 0

output = []

print(f"Evaluating {len(test_set)} samples from {dataset}")

with torch.no_grad():
    for item in tqdm(test_set):
        target_text = item["phoneme"]
        target_segs = ft.ipa_segs(target_text)

        if not target_segs:
            continue

        if args.epitran:
            raw_sentence = item["sentence"].strip()
            raw_pred = epi.transliterate(raw_sentence)
            pred_text = normalize_characters(raw_pred)
        elif args.base_model:
            raw_sentence = item["sentence"].strip()
            words = raw_sentence.split()

            predicted_words = []
            for word in words:
                input_text = f"<tgl>: {word}"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=50)
                decoded_word = tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_words.append(normalize_characters(decoded_word))

            pred_text = " ".join(predicted_words)
        else:
            inputs = tokenizer(item["sentence"], return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=256)
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred_segs = ft.ipa_segs(pred_text)

        # print("-" * 80)
        # print(f"Target:  {target_text}\nPredict: {pred_text}")

        # Calculate PER distance (does not include normalization yet)
        per_dist = dst.levenshtein_distance(pred_segs, target_segs)

        # Calculate PFER distance (we need a try-catch in case Panphon alignment dies)
        try:
            pfer_dist = dst.feature_edit_distance(pred_text, target_text)
        except ValueError:
            pfer_dist = len(target_segs)

        # Calculate CER distance (Levenshtein on raw characters)
        cer_dist = dst.levenshtein_distance(pred_text, target_text)

        # Add to the "pooled" summation
        # For corpus-level PER/CER/PFER, this is used to normalize by entire
        # corpus length
        total_per_dist += per_dist
        total_pfer_dist += pfer_dist
        total_cer_dist += cer_dist
        total_phonemes += len(target_segs)
        total_chars += len(target_text)

        # running_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
        # print(f"Running PER: {running_per}")

        # For sample-level PER/CER/PFER, normalize by sample lengths
        output.append(
            {
                "sentence": item["sentence"],
                "target": target_text,
                "predicted": pred_text,
                "per": per_dist / len(target_segs),
                "cer": cer_dist / len(target_text) if len(target_text) > 0 else 0,
                "pfer": pfer_dist / len(target_segs),
            }
        )

# Pooled statistics
final_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
final_pfer = total_pfer_dist / total_phonemes if total_phonemes > 0 else 0
final_cer = total_cer_dist / total_chars if total_chars > 0 else 0

print("=" * 40)
print("Pooled error statistics")
print("-" * 40)
print(f"Character Error Rate (CER):         {final_cer:.4f}")
print(f"Phoneme Error Rate (PER):           {final_per:.4f}")
print(f"Phonetic Feature Error Rate (PFER): {final_pfer:.4f}")
print(f"Total reference phonemes:           {total_phonemes}")

# Convert per-sample results to Pandas
df = pd.DataFrame(output)

print("-" * 40)
print("Per-sample error statistics")
print("-" * 40)
print(df[["per", "cer", "pfer"]].describe())

if args.epitran:
    df.to_pickle(f"results/output_epitran_{args.dataset}.pkl")
elif args.base_model:
    df.to_pickle(f"results/output_base_{args.dataset}.pkl")
else:
    df.to_pickle(
        f"results/output_{Path(args.checkpoint_path).parts[-2]}_{Path(args.checkpoint_path).parts[-1]}_{args.dataset}.pkl"
    )

print("=" * 40)
print("LaTeX table row output")
print("-" * 40)

latex_row = (
    f"& {final_per * 100:.2f} & {df['per'].mean() * 100:.2f} & {df['per'].std() * 100:.2f} "
    f"& {final_cer * 100:.2f} & {df['cer'].mean() * 100:.2f} & {df['cer'].std() * 100:.2f} "
    f"& {final_pfer * 100:.2f} & {df['pfer'].mean() * 100:.2f} & {df['pfer'].std() * 100:.2f} \\\\"
)

print(latex_row)

print("=" * 40)
print("Top 20 worst PER")
print("-" * 40)

for _, row in df.nlargest(20, "per").iterrows():
    print(f"PER:      {row['per']:.4f}")
    print(f"Sentence: {row['sentence']}")
    print(f"Target:   {row['target']}")
    print(f"Predict:  {row['predicted']}")
    print("-" * 40)
