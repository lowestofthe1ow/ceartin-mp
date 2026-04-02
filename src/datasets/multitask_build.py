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

    _, non_homo = wikipron_tl_df(str(wikipron_tsv))

    wp_data = []
    for word, pron in non_homo.items():
        wp_data.append({"sentence": f"{word}", "phoneme": pron})

    df_wikipron = pd.DataFrame(wp_data)
    df_wikipron["phoneme"] = df_wikipron["phoneme"].apply(normalize_characters)

    word_rows = df_wikipron["sentence"]
    repeated_words = word_rows[word_rows.duplicated(keep=False)]
    print(f"Found {len(repeated_words)} repeated 'G2P word' entries")

    df_wikipron.to_csv(output_path)

    print(f"Total rows: {len(df_wikipron)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    build_multitask_dataset()
