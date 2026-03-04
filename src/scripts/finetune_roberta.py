# Stuff changed: 
# changed Trainer to Seq2SeqTrainer and TrainingArguments to Seq2SeqTrainingArguments
# changed model to encoderdecodermodel (idk if decoder model is same as encoder model)
# changed datacollator to DataCollatorForSeq2Seq
# preprocess_nli uses encoder decoder format

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.optimization import Adafactor

from datasets import load_dataset

model_name = "jcblaise/roberta-tagalog-base"
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name) # idk if encoder model and decoder model are same
tokenizer = AutoTokenizer.from_pretrained(model_name)

# decoder config stuff 
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

# ------------------------------------------------------------------------------
# Load the dataset
# TODO: This is temporary! We won't actually be working with NewsPH-NLI


def preprocess_nli(examples):
    """Tokenizes NewsPH-NLI"""

    inputs = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=128
    )

    # changing label to text 
    targets = ["entailment" if l == 1 else "contradiction"
               for l in examples["label"]]

    # tokenize text
    labels = tokenizer(
        text_target=targets,
        truncation=True,
        max_length=8,
    )

    # attaching label to input
    inputs["labels"] = labels["input_ids"]

    return inputs



dataset = load_dataset("jcblaise/newsph_nli")
tokenized_dataset = dataset.map(preprocess_nli, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, model=model)
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

training_args = Seq2SeqTrainingArguments(
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

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(50000)),
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    optimizers=(optimizer, None),  # Pass custom Adafactor
)

trainer.train()
