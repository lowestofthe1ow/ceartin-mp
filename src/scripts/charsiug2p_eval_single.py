"""
This script performs a single evaluation of a given sentence on a trained model
checkpoint.
"""

import argparse

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

DEFAULT_SENTENCE = "Puno ng tubig ang baso."
DEFAULT_CHECKPOINT_PATH = "./models/checkpoints/checkpoint-728"

parser = argparse.ArgumentParser()
parser.add_argument("sentence", nargs="?", default=DEFAULT_SENTENCE)
parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained(args.checkpoint_path).to(device)


def predict(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


print(predict(args.sentence))
