from string import Template

# TODO: This is the current working prompt template...
TEMPLATE_STR = """<sentence>
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
   Tagalog. Be wary of repetition.
3. Inference: For words marked [NO CHOICES], infer the IPA transcription based
   on standard Filipino pronunciation.
4. Formatting rules:
    - Include stress markers (') at the beginning of the stressed syllable.
    - Exclude prosodic markers (like tone) and syllable separators (dots).
    - Provide the response as a direct, comma-separated list of IPA strings in
    the exact order the words appear in the sentence.
</instructions>

<word_choices>
$pronunciations
</word_choices>
"""

# TODO: Possible alternative prompt template
"""You are a grapheme-to-phoneme conversion system. The Tagalog word/s:

$words

in the sentence:

$sentence

has/have the following possible pronunciations:

$pronunciations

Select which pronunciations are best given the context of the sentence from the
options provided. Make sure your response matches the selected choice perfectly,
but only if the absolute correct prononciation is among the options. Otherwise,
for words with [ROOT], infer the pronunciation of the FULL word, including
inflections, based on the behavior shown in the choices for the root word
(remember that stress can shift with inflections). For words with [NO CHOICES],
infer the pronunciation. An apostrophe (') denotes the beginning of a stressed
syllable.Exclude prosodic information except stress. Exclude syllable separators
(dots) in your response. Your response must be direct and contain only your
selections for each word in the same order they occur in the given sentence,
separated by commas.
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
