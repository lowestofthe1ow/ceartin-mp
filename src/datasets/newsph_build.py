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
train_df = train_data.to_pandas()


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

    # Filter out short strings
    df_final = df_final[df_final["text"].str.split().str.len() > 20]

    # Filter out common English words to minimize Taglish
    df_final = df_final[
        ~df_final["text"].str.contains(
            ENG_STOPWORDS_RE, case=False, na=False, regex=True
        )
    ]

    # Remove all numbers
    df_final = df_final[~df_final["text"].str.contains(r"\d", na=False)]

    # Remove all sentences without homographs
    df_final = df_final[
        df_final["text"].apply(lambda x: any(w in HOMOGRAPHS_SET for w in x.split()))
    ]

    # Remove all sentences with fully capitalized words
    df_final = df_final[
        ~df_final["text"].apply(lambda x: any(word.isupper() for word in x.split()))
    ]

    # Remove sentences with certain symbols
    df_final = df_final[
        ~df_final["text"].str.contains(r"[\"'“”‘’@#$%^&*()_+=<>\[\]{}]")
    ]

    # Remove all empty strings just in case...
    df_final = df_final[df_final["text"].fillna("").str.strip() != ""]

    # Strip all strings
    df_final["text"] = df_final["text"].str.strip()

    # Sample 50k sentences
    df_final = df_final.sample(n=50000, random_state=765).reset_index(drop=True)

    return df_final


train_df = process_newsph_nli(train_df)

print(train_df.info())
print(train_df.head(5))
print(train_df["text"][0])

os.makedirs("data/newsph-nli", exist_ok=True)
train_df["text"].to_csv(
    "data/newsph-nli/newsph-nli.txt", index=False, header=False, sep="\n"
)
