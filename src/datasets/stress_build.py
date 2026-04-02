import argparse
import json

import pandas as pd
from tqdm import tqdm


# Helper to filter, format, and strip quotes
def process_sentence(s):
    # Remove all quotation marks immediately
    s = s.replace('"', "").replace("'", "").strip()
    if s.count(".") > 1:
        return None
    return s if s.endswith(".") else f"{s}."


DEFAULT_DATASET_PATH = "data/stress-minimal/results_gemini_2.5.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
parser.add_argument("--output-single", type=str, default="output_sin.txt")
parser.add_argument("--output-ambiguous", type=str, default="output_amb.txt")
args = parser.parse_args()

single_rows, ambiguous_rows = [], []

with open(args.dataset_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing JSONL"):
        data = json.loads(line)
        if data.get("success") and data.get("content"):
            word = data["content"]["word"].strip("'")
            valid_answers = []

            for ans in data["content"]["answers"]:
                cleaned = process_sentence(ans["sentence"])
                if cleaned:
                    valid_answers.append({"p": ans["pronunciation"], "s": cleaned})

            if not valid_answers:
                continue

            # Determine split based on unique pronunciations
            target = (
                single_rows
                if len(set(a["p"] for a in valid_answers)) == 1
                else ambiguous_rows
            )
            for a in valid_answers:
                target.append({"word": word, "sentence": a["s"]})


def save_to_txt(rows, path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Group by word and join sentences with a single space
    final_sentences = df.groupby("word")["sentence"].apply(" ".join)
    # Write each concatenated line to the text file
    with open(path, "w", encoding="utf-8") as f:
        for line in final_sentences:
            f.write(line + "\n")


save_to_txt(single_rows, args.output_single)
save_to_txt(ambiguous_rows, args.output_ambiguous)

print(f"Done. Created {args.output_single} and {args.output_ambiguous}")
