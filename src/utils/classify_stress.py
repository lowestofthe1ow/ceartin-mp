import csv
import sys
from collections import defaultdict

VOWELS = "aeiouəʌɛɪʊ"
CLASSES = ["Malumay", "Malumi", "Mabilis", "Maragsa", "None"]

"""
    Malumay - 'buhay - stress on penult
    Malumi - 'pasoʔ - stress on penult + ends in glottal stop
    Mabilis - bu'hay - stress on ult
    Maragsa - pa'soʔ - stress on ult + ends in glottal stop
"""


def syllabicate(phonetic_word):
    """Splits a Tagalog IPA string into syllables"""

    tokens = list(phonetic_word)
    syllables, current = [], ""
    i = 0
    while i < len(tokens):
        ch = tokens[i]
        if ch == "'":
            if current:
                syllables.append(current)
            current = ch
            i += 1
            continue
        current += ch
        if ch in VOWELS:
            j = i + 1
            while j < len(tokens):
                ahead = tokens[j]
                if ahead == "'" or ahead in VOWELS:
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


def classify_stress(phonetic_word):
    """Classifies a word into one of four stress classes"""

    if not phonetic_word or not isinstance(phonetic_word, str):
        return "None"  # default
    if "'" not in phonetic_word:
        return "None"
    has_glottal = phonetic_word.endswith("ʔ")
    syllables = syllabicate(phonetic_word)
    if not syllables:
        return "Malumay"
    is_last_stressed = "'" in syllables[-1]
    if is_last_stressed:
        return "Maragsa" if has_glottal else "Mabilis"
    else:
        return "Malumi" if has_glottal else "Malumay"


def compute_metrics(y_true, y_pred):
    """Calculates precision, recall, F1 per class"""

    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for g, p in zip(y_true, y_pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    results = {}
    for cls in CLASSES:
        t, f_p, f_n = tp[cls], fp[cls], fn[cls]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[cls] = {"Precision": prec, "Recall": rec, "F1": f1, "Support": t + f_n}
    return results


def run_evaluation(csv_path: str):
    y_true, y_pred, table_rows = [], [], []

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, 1):
                # sentence-to-word breakdown
                # 'meɾon ka 'baŋ -> ['meɾon, ka, 'baŋ]
                target_words = str(row.get("target", "")).split()
                pred_words = str(row.get("predicted", "")).split()

                # word-for-word comparison part thru zip
                for t, p in zip(target_words, pred_words):
                    tc = classify_stress(t)
                    pc = classify_stress(p)
                    match = "yes" if tc == pc else "no"

                    # [ (row, target, t_class, pred, p_class, match), ... ]
                    table_rows.append((row_idx, t, tc, p, pc, match))
                    y_true.append(tc)
                    y_pred.append(pc)
        metrics = compute_metrics(y_true, y_pred)
        print(metrics)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return


print(classify_stress("mag'kabilaŋ"))

run_evaluation("output.csv")
