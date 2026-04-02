import re
from pathlib import Path

import pandas as pd

from src.datasets.wikipron_tl_df import wikipron_tl_df

# For some reason it errors when i import this
# from src.scripts.read_jsonl import normalize_characters


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


def build_multitask_dataset():
    wikipron_tsv = "data/wikipron/wikipron_tl.tsv"
    output_path = "data/multitask/multitask_ds.csv"

    newsph_df = pd.read_csv("data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv")
    tatoeba_df = pd.read_csv("data/tatoeba/phonetic_tatoeba_gemini_3.csv")
    newsph_df["sentence"] = "G2P sentence: " + newsph_df["sentence"].astype(str)
    tatoeba_df["sentence"] = "G2P sentence: " + tatoeba_df["sentence"].astype(str)

    _, non_homo = wikipron_tl_df(str(wikipron_tsv))

    wp_data = []
    for word, pron in non_homo.items():
        wp_data.append({"sentence": f"G2P word: {word}", "phoneme": pron})

    df_wikipron = pd.DataFrame(wp_data)
    df_wikipron["phoneme"] = df_wikipron["phoneme"].apply(normalize_characters)

    multitask_df = pd.concat(
        [
            newsph_df[["sentence", "phoneme"]],
            tatoeba_df[["sentence", "phoneme"]],
            df_wikipron,
        ],
        ignore_index=True,
    )

    multitask_df = multitask_df.sample(frac=1, random_state=961).reset_index(drop=True)
    multitask_df.to_csv(output_path, index=False)

    word_rows = multitask_df[multitask_df["sentence"].str.startswith("G2P word: ")]
    repeated_words = word_rows[word_rows.duplicated(subset=["sentence"], keep=False)]
    print(f"Found {len(repeated_words)} repeated 'G2P word' entries")

    sentence_rows = multitask_df[
        multitask_df["sentence"].str.startswith("G2P sentence: ")
    ]
    repeated_sentences = sentence_rows[
        sentence_rows.duplicated(subset=["sentence"], keep=False)
    ]
    print(f"Found {len(repeated_sentences)} repeated 'G2P sentence' entries")
    print(repeated_sentences)

    print("Removing repeated 'G2P sentence' entries...")
    multitask_df.drop_duplicates(subset=["sentence"], keep="first", inplace=True)
    print(f"Total rows: {len(multitask_df)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    build_multitask_dataset()
