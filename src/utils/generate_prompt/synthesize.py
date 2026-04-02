from string import Template

from dotenv import dotenv_values

from src.datasets.wikipron_tl_df import wikipron_tl_df

config = dotenv_values(".env")

FILE_PATH = config["WIKIPRON_PATH"]

HOMOGRAPHS, _ = wikipron_tl_df(FILE_PATH)

TEMPLATE_STR = """<instructions>
The Filipino word given below has several possible pronunciations, given in a
list. For each unique pronunciation, generate sentences that use the word
following the definition that corresponds to the pronunciation. The objective
is to create a list of sentences (in Filipino, not in IPA) that use the same
word in different meanings or contexts, four for each definition corresponding
to the pronunciations listed. Follow standard Filipino grammar. Exclude all
diacritics from your responses. You may perform conjugations if applicable.
Include the word itself and the index of its intended pronunciation in the list,
where the first entry is index 1. If any of the pronunciations listed are simply
variations and do not SIGNIFICANTLY change the meaning of the word, treat them
as a single pronunciation and generate four sentences using only the FIRST
pronunciation listed that has that specific meaning. Only in this case are you
to ignore the other variants. Otherwise, generate four sentences for each
pronunciation listed as instructed. Each sentence must be a separate entry
string in the lists of sentences in the output schema.
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
