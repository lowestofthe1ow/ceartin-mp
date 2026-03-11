#!/bin/bash

OUT_PATH="data/phonetic_tatoeba/"
mkdir -p "$OUT_PATH"

IDS=(
    "1PcEsHXjc7xeyNJI17F1KDOcLhdei-VAM"
    "1gq52lpqfjEF2wI1bFo7ddrU-Hmg6rEAu"
    "1E0XmtGAXP1DZnFrGnCtijcKnbRfcM-Vk"
    "1MySvgmI22kgunOKCncvZttKO9GYxVtDZ"
)

FILE_NAMES=(
    "phonetic_tatoeba_gemini_2.5.csv"
    "phonetic_tatoeba_gemini_3.csv"
    "results_gemini_2.5.jsonl"
    "results_gemini_3.jsonl"
)

for i in "${!IDS[@]}"; do
    ID=${IDS[$i]}
    NAME=${FILE_NAMES[$i]}
    URL="https://drive.google.com/uc?export=download&id=$ID"

    wget -O "$OUT_PATH$NAME" "$URL"

    echo "Wrote to file $OUT_PATH$FILE_NAME."
done
