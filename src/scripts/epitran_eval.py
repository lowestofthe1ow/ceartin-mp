import argparse
import csv
import re

import epitran
import panphon
import panphon.distance
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils.dataset_from_csv import dataset_from_csv_list


# Crashes when I try to import the function so uh here
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="tatoeba",
    choices=["tatoeba", "newsph-nli", "combined", "multitask"],
)
args = parser.parse_args()

if args.dataset == "tatoeba":
    dataset = ["data/tatoeba/phonetic_tatoeba_gemini_3.csv"]
elif args.dataset == "newsph-nli":
    dataset = ["data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv"]
elif args.dataset == "combined":
    dataset = [
        "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
        "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    ]
elif args.dataset == "multitask":
    dataset = ["data/multitask/multitask_ds.csv"]

tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_ID)

epi = epitran.Epitran("tgl-Latn")

ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

split_dataset = dataset_from_csv_list(dataset, tokenizer)
test_set = split_dataset["test"]

total_per_dist = 0
total_pfer_dist = 0
total_cer_dist = 0
total_phonemes = 0
total_chars = 0
sample_pers = []

print(f"Evaluating Epitran on {len(test_set)} samples from {args.dataset}")

out_file = open(f"output_epitran_{args.dataset}.csv", "w", encoding="utf-8", newline="")
csv_writer = csv.writer(out_file)
csv_writer.writerow(["target", "predicted"])

for item in tqdm(test_set):
    target_text = item["phoneme"]
    target_segs = ft.ipa_segs(target_text)

    if not target_segs:
        continue

    clean_sentence = item["sentence"].strip()

    pred_text = normalize_characters(epi.transliterate(clean_sentence))
    pred_segs = ft.ipa_segs(pred_text)
    csv_writer.writerow([target_text, pred_text])

    per_dist = dst.levenshtein_distance(pred_segs, target_segs)

    try:
        pfer_dist = dst.feature_edit_distance(pred_text, target_text)
    except ValueError:
        pfer_dist = len(target_segs)

    cer_dist = dst.levenshtein_distance(pred_text, target_text)

    total_per_dist += per_dist
    total_pfer_dist += pfer_dist
    total_cer_dist += cer_dist

    total_phonemes += len(target_segs)
    total_chars += len(target_text)

    sample_per = per_dist / len(target_segs) if len(target_segs) > 0 else 0
    sample_pers.append((sample_per, clean_sentence, target_text, pred_text))

final_per = total_per_dist / total_phonemes if total_phonemes > 0 else 0
final_pfer = total_pfer_dist / total_phonemes if total_phonemes > 0 else 0
final_cer = total_cer_dist / total_chars if total_chars > 0 else 0

print(f"Character Error Rate (CER):         {final_cer:.4f}")
print(f"Phoneme Error Rate (PER):           {final_per:.4f}")
print(f"Phonetic Feature Error Rate (PFER): {final_pfer:.4f}")
print(f"Total reference phonemes:           {total_phonemes}")
print("=" * 40)
print("Top 20 highest PER samples")
for sample_per, sentence, target_text, pred_text in sorted(sample_pers, reverse=True)[
    :20
]:
    print(f"PER: {sample_per:.4f} | Sentence: {sentence}")
    print(f"  Target:  {target_text}")
    print(f"  Predict: {pred_text}")

# Close the CSV file
out_file.close()
