from datasets import load_dataset

# 1. Load the dataset from Hugging Face
# This will download the dataset and cache it locally
dataset = load_dataset("jcblaise/newsph_nli")

# 2. Access the 'train' split
# The dataset typically contains 'train', 'validation', and 'test' splits
train_data = dataset["train"]

# 3. Print the first 5 samples
print(f"Printing the first 5 samples from the 'train' split:\n")
for i in range(5):
    sample = train_data[i]
    print(f"Sample {i+1}:")
    print(f"  Premise:    {sample['premise'].replace('"', "")}")
    print(f"  Hypothesis: {sample['hypothesis'].replace('"', "")}")
    print(
        f"  Label:      {sample['label']} (0: Entailment, 1: Neutral, 2: Contradiction)"
    )
    print("-" * 30)
