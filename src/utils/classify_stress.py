VOWELS = "aeiouəʌ"

def syllabicate(phonetic_word: str) -> list[str]:
    """
    Splits a Tagalog IPA string into syllables using onset-maximisation.

    Rules:
      - A stress marker (') begins a new syllable (attached to its onset).
      - Each syllable nucleus is a vowel.
      - A consonant between two vowels belongs to the RIGHT (onset of next syllable).
      - A consonant NOT followed by a vowel is a coda of the current syllable.
      - ʔ is treated as a consonant; when word-final it stays as coda of last syllable.
    """
    tokens = list(phonetic_word)
    syllables: list[str] = []
    current = ""

    i = 0
    while i < len(tokens):
        ch = tokens[i]

        if ch == "'":
            if current:
                syllables.append(current)
                current = ""
            current += ch
            i += 1
            continue

        current += ch

        if ch in VOWELS:
            j = i + 1
            while j < len(tokens):
                ahead = tokens[j]
                if ahead == "'":
                    break                      
                if ahead in VOWELS:
                    break                     
                k = j + 1
                while k < len(tokens) and tokens[k] == "'":
                    k += 1
                if k < len(tokens) and tokens[k] in VOWELS:
                    break                      
                current += ahead
                j += 1
            i = j
            syllables.append(current)
            current = ""
            continue

        i += 1

    if current:
        syllables.append(current)

    return syllables


def classify_accent(phonetic_word: str) -> str | None:
    """
    Classifies a phonetic word into one of four Tagalog accent classes.

    Malumay  — stress on penultimate syllable, does NOT end in ʔ
    Malumi   — stress on penultimate syllable, ends in ʔ
    Mabilis  — stress on last syllable,        does NOT end in ʔ
    Maragsa  — stress on last syllable,        ends in ʔ

    Returns None for unstressed function words (no stress marker present).
    #TODO: either drop this if not needed
    """
    if not phonetic_word:
        return None

    # Glottal stop check
    has_glottal = phonetic_word.endswith("ʔ")

    # Stress position — find the stress marker, then check whether any
    # vowel follows it (ignoring a word-final ʔ). If no vowel follows,
    # the stress is on the last syllable.
    stress_pos = phonetic_word.find("'")
    if stress_pos == -1:
        return None     # for no stress marker

    # check if any further vowel exists, indicating another syllable after.
    after_marker = phonetic_word[stress_pos + 1:]
    j = 0
    while j < len(after_marker) and after_marker[j] not in VOWELS:  # skip onset consonants
        j += 1
    while j < len(after_marker) and after_marker[j] in VOWELS:      # skip nucleus vowels
        j += 1
    remainder = after_marker[j:].replace("ʔ", "")                   # strip word-final ʔ
    is_last_stressed = not any(c in VOWELS for c in remainder)

    # 2×2 classification
    if is_last_stressed:
        return "Maragsa" if has_glottal else "Mabilis"
    else:
        return "Malumi" if has_glottal else "Malumay"