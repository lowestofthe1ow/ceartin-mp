import pandas as pd

from src.scripts.read_jsonl import normalize_characters


def process_csv(file_path, output_path: str = "normalized_output.csv"):
    df = pd.read_csv(file_path)[["sentence", "phoneme"]].fillna("")
    df["phoneme"] = df["phoneme"].astype(str).apply(normalize_characters)
    df.to_csv(output_path, index=False)


process_csv("data/manual_set.csv")
