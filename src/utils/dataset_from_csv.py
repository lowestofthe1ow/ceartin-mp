from datasets import DatasetDict, load_dataset

RANDOM_STATE = 765  # ナムコプロ最強


def preprocess_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        text_target=examples["phoneme"],
        max_length=128,  # TODO: Confirm if we're gonna use this limit
        padding="max_length",  # Forces everything to have the same length
        truncation=True,  # This truncates when it generates > max_length
    )


def dataset_from_csv(csv_path, tokenizer):
    dataset = load_dataset("csv", data_files=csv_path)["train"]
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )

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

    return split_dataset
