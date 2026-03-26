from transformers import AutoTokenizer

from src.utils.dataset_from_csv import dataset_from_csv_list

# TODO: Yes I know this whole thing is ugly but whatever lol

MODEL_ID = "charsiu/g2p_multilingual_byT5_small_100"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dataset_files = [
    "data/tatoeba/phonetic_tatoeba_gemini_3.csv",
    "data/newsph-nli/phonetic_newsph-nli_gemini_2.5_lite.csv",
    "data/stress-minimal/stress-minimal_ambiguous_split.csv",
    "data/stress-minimal/stress-minimal_single_split.csv",
]

split_dataset = dataset_from_csv_list(dataset_files, tokenizer)

test_split = split_dataset["test"]
print(f"Original test set size: {len(test_split)} rows")

sampled_test_dict = test_split.train_test_split(test_size=0.1695, seed=765)
sampled_test_10_percent = sampled_test_dict["test"]

columns_to_keep = ["index", "sentence", "phoneme"]
sampled_test_10_percent = sampled_test_10_percent.select_columns(columns_to_keep)

print(f"Sampled size: {len(sampled_test_10_percent)} rows")

output_path = "manual.csv"
sampled_test_10_percent.to_csv(output_path, index=False)
