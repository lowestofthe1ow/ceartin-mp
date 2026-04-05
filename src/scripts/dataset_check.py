# TODO: We don't use all these imports but whatever
import argparse
from datetime import datetime

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from datasets import concatenate_datasets
from src.utils.dataset_from_csv import dataset_from_csv, dataset_from_csv_list
from src.utils.preview_callback import PreviewCallback

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = "data/combined.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_id)

dataset = [
    "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split.csv",
    "data/stress-minimal/stress-minimal_single_split.csv",
]

split_dataset = dataset_from_csv_list(dataset, tokenizer)

train_set = split_dataset["train"]
manual_set = dataset_from_csv("data/manual_set.csv", tokenizer)
manual_set = concatenate_datasets(list(manual_set.values()))

train_sentences = set(train_set["sentence"])
manual_sentences = set(manual_set["sentence"])

leakage = train_sentences.intersection(manual_sentences)

if len(leakage) > 0:
    print(f"Found {len(leakage)} leaking examples")
    for i, sentence in enumerate(list(leakage)[:5]):
        print(f"{i+1}. {sentence}")
else:
    print("No leakage detected")
