"""
This script begins a finetune of the charsiu/g2p_multilingual_byT5_small_100
model using the HuggingFace API.
"""

# TODO: Resolve the following warning (?):
# "The tied weights mapping and config for this model specifies to tie
# shared.weight to decoder.embed_tokens.weight, but both are present in the
# checkpoints, so we will NOT tie them. You should update the config with
# `tie_word_embeddings=False` to silence this warning."

import argparse

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from src.utils.dataset_from_csv import dataset_from_csv
from src.utils.preview_callback import PreviewCallback

RANDOM_STATE = 765  # ナムコプロ最強

DEFAULT_MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
DEFAULT_DATASET_PATH = "data/phonetic_tatoeba/phonetic_tatoeba_gemini_3.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
split_dataset = dataset_from_csv(args.dataset_path, tokenizer)

# Set up model and DataCollator
model = T5ForConditionalGeneration.from_pretrained(args.model_id)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Optimizer and trainer configuration

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/checkpoints",
    bf16=True,
    optim="adafactor",
    # --------------------------------------------
    # Effective batch size: 16 * 4 = 64
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    # --------------------------------------------
    # Use standard ByT5 learning rates...
    learning_rate=3e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=1000,
    # --------------------------------------------
    num_train_epochs=10,  # TODO: Is 10 good?
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    logging_steps=10,
    # --------------------------------------------
    # Max generated length of 512 for sentences
    predict_with_generate=True,
    generation_max_length=512,
    # --------------------------------------------
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["validation"],  # type: ignore
    data_collator=data_collator,
    callbacks=[PreviewCallback()],
)

trainer.train()
