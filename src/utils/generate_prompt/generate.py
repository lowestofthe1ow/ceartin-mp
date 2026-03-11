from string import Template

from dotenv import dotenv_values

from src.datasets.wikipron.wikipron_tl_df import wikipron_tl_df

config = dotenv_values(".env")

FILE_PATH = config["WIKIPRON_PATH"]

HOMOGRAPHS, _ = wikipron_tl_df(FILE_PATH)

TEMPLATE_STR = """<instructions>
The Filipino word given below has several possible pronunciations, given in a
list. For each unique pronunciation, generate a sentence that uses the word
following the definition that corresponds to the pronunciation. The objective
is to create a list of sentences (in Filipino, not in IPA) that use the word
differently, one for each pronunciation. Follow standard Filipino grammar.
Exclude all diacritics from your responses.
</instructions>

<word>$word</word>

<pronunciations_list>
$pronunciations
</pronunciations_list>
"""

TEMPLATE = Template(TEMPLATE_STR)


def generate_prompt(word):
    return TEMPLATE.safe_substitute(
        word=word, pronunciations="\n".join(HOMOGRAPHS.get(word))
    )
