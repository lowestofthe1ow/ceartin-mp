import json

from tqdm import tqdm

FILE_PATH = "results_gemini_3.jsonl"

sentences = []

with open(FILE_PATH, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="Processing JSONL", unit="line"):

        data = json.loads(line)

        if data.get("success") is True:
            answers = data["content"]["answers"]

            sentences = [*sentences, *answers]


print(len(sentences))
