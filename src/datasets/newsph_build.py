import os
import re

import pandas as pd
from dotenv import dotenv_values

from datasets import load_dataset
from src.datasets.wikipron_tl_df import wikipron_tl_df

config = dotenv_values(".env")

FILE_PATH = config["WIKIPRON_PATH"]

HOMOGRAPHS, NON_HOMOGRAPHS = wikipron_tl_df(FILE_PATH)
HOMOGRAPHS_SET = set(HOMOGRAPHS.keys())

dataset = load_dataset("jcblaise/newsph_nli")
train_data = dataset["train"]

df = train_data.to_pandas()


def process_newsph_nli(df):
    # Combine both the premise and the hypothesis
    combined_text = pd.concat([df["premise"], df["hypothesis"]], ignore_index=True)
    df_final = pd.DataFrame(combined_text, columns=["text"])

    ENG_STOPWORDS = [
        "the",
        "of",
        "and",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "with",
        "as",
        "you",
        "do",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
    ]

    ENG_STOPWORDS_RE = (
        r"\b(" + "|".join(re.escape(word) for word in ENG_STOPWORDS) + r")\b"
    )

    df_final = df_final[df_final["text"].str.split().str.len() > 20]
    df_final = df_final[
        ~df_final["text"].str.contains(
            ENG_STOPWORDS_RE, case=False, na=False, regex=True
        )
    ]
    df_final = df_final[~df_final["text"].str.contains(r"\d", na=False)]

    df_final = df_final[
        df_final["text"].apply(lambda x: any(w in HOMOGRAPHS_SET for w in x.split()))
    ]

    df_final = df_final[
        ~df_final["text"].apply(lambda x: any(word.isupper() for word in x.split()))
    ]

    df_final = df_final[
        ~df_final["text"].str.contains(r"[\"'“”‘’@#$%^&*()_+=<>\[\]{}]")
    ]
    df_final = df_final[df_final["text"].fillna("").str.strip() != ""]
    df_final["text"] = df_final["text"].str.strip()

    df_final = df_final.sample(n=50000, random_state=765).reset_index(drop=True)

    return df_final


df = process_newsph_nli(df)

print(df.info())

print(df.head(5))

print(df["text"][0])

os.makedirs("data/newsph-nli", exist_ok=True)
df["text"].to_csv("data/newsph-nli/newsph-nli.txt", index=False, header=False, sep="\n")

"""
# 3. Print the first 5 samples
print(f"Printing the first 5 samples from the 'train' split:\n")
for i in range(5):
    sample = train_data[i]
    print(f"Sample {i+1}:")
    print(f"  Premise:    {sample['premise'].replace('"', "")}")
    print(f"  Hypothesis: {sample['hypothesis'].replace('"', "")}")
    print(
        f"  Label:      {sample['label']} (0: Entailment, 1: Neutral, 2: Contradiction)"
    )
    print("-" * 30)
"""
