#!/bin/bash

# WikiPron data
source "bash/get_wikipron.sh"

# PhoneticTatoeba data
source "bash/get_phonetic_tatoeba.sh"

# Tatoeba data
uv run -m src.datasets.tatoeba.make_file
