"""
This script evaluates a model checkpoint on a given test set by calculating PER
and PFER.
"""

import argparse

import panphon
import panphon.distance
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.dataset_from_csv import dataset_from_csv

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_CHECKPOINT = "./models/checkpoints/checkpoint-728"
DEFAULT_DATASET_PATH = "data/phonetic_tatoeba/phonetic_tatoeba_gemini_3.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
args = parser.parse_args()

# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path)
model.to(device)
model.eval()

# Use Panphon's built in tools, less of a headache this way
ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

# Get the test split
split_dataset = dataset_from_csv(args.dataset_path, tokenizer)
test_set = split_dataset["test"]

# Running sums
total_per_dist = 0
total_pfer_dist = 0
total_phonemes = 0

print(f"Evaluating {len(test_set)} samples from {args.dataset_path}")

with torch.no_grad():
    for item in tqdm(test_set):
        target_text = item["phoneme"]
        target_segs = ft.ipa_segs(target_text)

        if not target_segs:
            continue

        inputs = tokenizer(item["sentence"], return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=512)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_segs = ft.ipa_segs(pred_text)

        # Calculate PER
        per_dist = dst.levenshtein_distance(pred_segs, target_segs)

        # Calculate PFER (we need a try-catch in case Panphon alignment dies)
        try:
            pfer_dist = dst.feature_edit_distance(pred_text, target_text)
        except ValueError:
            pfer_dist = len(target_segs)

        # Add to the summation
        total_per_dist += per_dist
        total_pfer_dist += pfer_dist
        total_phonemes += len(target_segs)

final_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
final_pfer = total_pfer_dist / total_phonemes if total_phonemes > 0 else 0

print(f"Phoneme Error Rate (PER):           {final_per:.4f}")
print(f"Phonetic Feature Error Rate (PFER): {final_pfer:.4f}")
print(f"Total reference phonemes:           {total_phonemes}")
