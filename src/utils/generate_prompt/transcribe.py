from string import Template

# This list was obtained from WikiPron's scraped data.
PHONEME_INVENTORY = [
    "'",
    "a",
    "b",
    "d",
    "e",
    "f",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "ŋ",
    "ɕ",
    "ə",
    "ɡ",
    "ɹ",  # NOTE: Only used by one word
    "ɾ",
    "ʃ",
    "ʌ",
    "ʒ",
    "ʔ",
    "ˈ",  # NOTE: Technically a "duplicate"
    "ˌ",
    # NOTE: We handle the tie bar with a special instruction
    " ‍͡ (Tie bar: Unicode: U+0361)",
]

# TODO: This is the current working prompt template...
TEMPLATE_STR = f"""<sentence>
$sentence
</sentence>

<instructions>
Provide the most accurate IPA transcription for the listed words given the
context in the provided sentence, ensuring stress placement (') is correct based
on Tagalog grammar and context.
1. Selection criteria: If the correct pronunciation is in the provided options,
   select it exactly.
2. Root & inflection handling: For words marked [ROOT], infer the IPA for the
   FULL inflected word as it appears in the sentence, based on the provided
   root. Note that stress typically shifts when certain suffixes are added in
   Tagalog. Be careful not to accidentally merge repeated sounds. In the choices
   provided, note that the stress marker (') is included at the start of the
   stressed syllable.
3. Inference: For words marked [NO CHOICES], infer the IPA transcription based
   on standard Filipino pronunciation.
4. Formatting rules:
    - Only include answers for the list of words given.
    - Place stress markers (') at the BEGINNING of the stressed syllable.
    - Exclude prosodic markers (like tone) and syllable separators (dots).
    - Provide the response as a direct, comma-separated list of IPA strings in
    the exact order the words appear in the sentence.
    - You may use only the UTF-8 characters provided in the pphoneme inventory.
</instructions>

<phoneme_inventory>
{PHONEME_INVENTORY}
</phoneme_inventory>

<word_list>
$pronunciations
</word_list>
"""


TEMPLATE = Template(TEMPLATE_STR)


def _format_word_data(data):
    choices = data.get("choices")

    if choices:
        # If word was found in WikiPron scrape...
        pron_str = ", ".join(choices)
    else:
        # Otherwise, fall back to root data
        root = data.get("root", {})
        root_word = root.get("word", "")
        root_prons = ", ".join(root.get("choices", []))
        pron_str = f'[ROOT "{root_word}"] {root_prons}'

    return f"{data.get("word")}: {pron_str}"


def generate_prompt(sentence, pronunciations, words):
    # Comma-separated, with each word in quotes
    formatted_words = ", ".join([f'"{word}"' for word in words])

    # Show choices for each ambiguous word
    formatted_pronunciations = "\n".join(
        _format_word_data(data) for data in pronunciations
    )

    # Wrap in quotes
    formatted_sentence = f'"{sentence}"'

    return TEMPLATE.safe_substitute(
        words=formatted_words,
        sentence=formatted_sentence,
        pronunciations=formatted_pronunciations,
    )
