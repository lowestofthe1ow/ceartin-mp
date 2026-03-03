from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.optimization import Adafactor

from datasets import load_dataset

model_name = "jcblaise/roberta-tagalog-base"
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ------------------------------------------------------------------------------
# Load the dataset
# TODO: This is temporary! We won't actually be working with NewsPH-NLI


def preprocess_nli(examples):
    """Tokenizes NewsPH-NLI"""
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


dataset = load_dataset("jcblaise/newsph_nli")
tokenized_dataset = dataset.map(preprocess_nli, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# ------------------------------------------------------------------------------

# We define the Adafactor optimizer based on the following paper:
# https://arxiv.org/abs/2111.06053
# TODO: Check if I got everything right
optimizer = Adafactor(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.1,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)

training_args = TrainingArguments(
    output_dir="./models/checkpoints",
    # 8 physical batch size * 4 accumulation steps = 32
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    # Follow paper's scheduler configuration
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.1,
    weight_decay=0.1,
    # TODO: FP16?
    fp16=True,
    logging_steps=100,
    eval_strategy="epoch",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(50000)),
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    optimizers=(optimizer, None),  # Pass custom Adafactor
)

trainer.train()
