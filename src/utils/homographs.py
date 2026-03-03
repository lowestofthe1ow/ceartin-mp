import string

import pandas as pd
from dotenv import dotenv_values
from tglstemmer import stemmer

from src.datasets.wikipron.wikipron_tl_pre import wikipron_tl_df

config = dotenv_values(".env")

FILE_PATH = config["WIKIPRON_PATH"]

TRANSLATOR = str.maketrans("", "", string.punctuation)

HOMOGRAPHS, NON_HOMOGRAPHS = wikipron_tl_df(FILE_PATH)

# Convert to a hashed set for constant-time lookup
NON_HOMOGRAPHS_SET = set(NON_HOMOGRAPHS.keys())


def homographs(sentence):
    # Breaks down the sentence string into a list of words
    words = [word.lower().translate(TRANSLATOR).strip() for word in sentence.split()]

    # Identifies words with more than one pronunciation
    ambiguous_words = [word for word in words if word not in NON_HOMOGRAPHS_SET]

    # Get a list of all words in the sentence with homograhs replaced by "_"
    output_template = [NON_HOMOGRAPHS.get(word, "_") for word in words]

    # Get a list of the pronunciations of all homographs in the sentence
    choices = [
        {
            "word": word,
            # Check if there's a match...
            "choices": HOMOGRAPHS.get(word, []),
            # Otherwise, fall back to root word
            "root": {
                "word": (root := stemmer.get_stem(word)),
                # Check both homographs and non-homographs list
                "choices": HOMOGRAPHS.get(
                    root, [NON_HOMOGRAPHS.get(root, "[NO CHOICES]")]
                ),
            },
        }
        for word in words
        if word not in NON_HOMOGRAPHS_SET
    ]

    return ambiguous_words, choices, output_template


def fill_template(template, prons):
    prons_iter = iter(prons)
    output = [next(prons_iter) if item == "_" else item for item in template]
    return " ".join(output)
