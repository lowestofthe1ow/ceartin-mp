import json
import time
from typing import List

import pandas as pd
from dotenv import dotenv_values
from google import genai
from pydantic import BaseModel
from tqdm import tqdm

from datasets import load_dataset
from src.utils.generate_prompt import generate_prompt
from src.utils.homographs import homographs


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    answers: List[str]


# Set up Gemini API
config = dotenv_values(".env")
key = config["GEMINI_API_KEY"]
client = genai.Client(api_key=key)

# TODO: TLUnified dataset as alternative
# dataset = load_dataset("ljvmiranda921/tlunified-ner", split="train")
# sentences = [" ".join(item["tokens"]) for item in dataset]

# Load and process the Tatoeba dataset
dataset = load_dataset("tatoeba", "en-tl", lang1="en", lang2="tl")
sentences = [item["tl"] for item in dataset["train"]["translation"]]


def process_sentence(sentence):
    """Processes each sentence in Tatoeba and generates a prompt"""
    words, choices, _ = homographs(sentence)
    return generate_prompt(sentence=sentence, pronunciations=choices, words=words)


prompts = [process_sentence(s) for s in tqdm(sentences)]
# prompts = list(filter(None, prompts))

# Generate batch requests to Gemini
batch_requests = [
    {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "config": {
            "response_mime_type": "application/json",
            "response_json_schema": Response.model_json_schema(),
        },
    }
    for prompt in tqdm(prompts)
]

batch_job = client.batches.create(
    model="gemini-3-flash-preview",
    src=batch_requests,
    config={
        "display_name": "inlined-requests-job-1",
    },
)

print(f"Created batch job: {batch_job.name}.")
print("You may now exit this process, but it will poll the job's status.")

# Constantly poll the job status
while True:
    batch_job_inline = client.batches.get(name=batch_job.name)
    if batch_job_inline.state.name in (
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    ):
        break

    print(f"Job state: {batch_job_inline.state.name}. Waiting 30 seconds...")
    time.sleep(30)

# TODO: This part onwards could use some cleaning
# Process the final responses.
response_out = []
for i, inline_response in enumerate(batch_job_inline.dest.inlined_responses, start=1):
    result_entry = {"index": i, "success": False, "content": None, "error": None}

    if inline_response.response:
        result_entry["success"] = True
        result_entry["content"] = inline_response.response.text
    elif inline_response.error:
        # TODO: Captures API errors
        result_entry["error"] = str(inline_response.error)

    response_out.append(result_entry)

# Write to a JSON file.
output_filename = "batch_results.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(response_out, f, ensure_ascii=False, indent=4)

print(f"All results saved to {output_filename}")
