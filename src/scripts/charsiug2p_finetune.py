"""
This script begins a finetune of the charsiu/g2p_multilingual_byT5_small_100
model using the HuggingFace API, using the PhoneticTatoeba Tagalog dataset.
"""

# TODO: Resolve the following warning:
# "The tied weights mapping and config for this model specifies to tie
# shared.weight to decoder.embed_tokens.weight, but both are present in the
# checkpoints, so we will NOT tie them. You should update the config with
# `tie_word_embeddings=False` to silence this warning."

# TODO: Use argparse

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from datasets import DatasetDict, load_dataset

RANDOM_STATE = 765  # ナムコプロ最強
MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"


def preprocess_function(examples):
    return tokenizer(
        examples["sentence"],
        text_target=examples["phoneme"],
        max_length=128,  # TODO: Confirm if we're gonna use this limit
        padding="max_length",  # Forces everything to have the same length
        truncation=True,  # This truncates when it generates > max_length
    )


# Load the PhoneticTatoeba dataset
dataset = load_dataset(
    "csv", data_files="data/phonetic_tatoeba/phonetic_tatoeba_gemini_3.csv"
)["train"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Perform an 80%/10%/10% train-test-validation split
train_test = tokenized_dataset.train_test_split(test_size=0.2, seed=RANDOM_STATE)
val_test = train_test["test"].train_test_split(test_size=0.5, seed=RANDOM_STATE)

split_dataset = DatasetDict(
    {
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    }
)

# Set up model and DataCollator
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


class PreviewCallback(TrainerCallback):
    """Shows a sample evaluation at the end of a training epoch"""

    # TODO: This should be isolated into a module, maybe under utils

    def on_epoch_end(self, args, state, control, **kwargs):
        # Take 3 samples from the validation set
        samples = split_dataset["validation"].select(range(3))

        keys_to_keep = ["input_ids", "attention_mask", "labels"]
        batch_list = [
            {k: v for k, v in samples[i].items() if k in keys_to_keep}
            for i in range(len(samples))
        ]

        inputs = data_collator(batch_list)

        model_inputs = {
            "input_ids": inputs["input_ids"].to(args.device),
            "attention_mask": inputs["attention_mask"].to(args.device),
        }

        generated_tokens = kwargs["model"].generate(**model_inputs, max_length=128)

        preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        labels = tokenizer.batch_decode(inputs["labels"], skip_special_tokens=True)

        print("=" * 40)
        print(f"Epoch: {state.epoch}")
        print("-" * 40)
        for p, l in zip(preds, labels):
            print(f"Target:  {l}")
            print(f"Predict: {p}")
        print("=" * 40)


# Optimizer and trainer configuration
# TODO: Double-check configuration

training_args = Seq2SeqTrainingArguments(
    output_dir="./models/checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=3e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=100,
    dataloader_num_workers=0,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    predict_with_generate=True,
    generation_max_length=128,
    optim="adafactor",
    # Load the best model
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
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
