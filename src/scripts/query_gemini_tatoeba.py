"""
This script concurrently sends a series of requests to the Gemini API to build
the PhoneticTatoeba dataset.
"""

import asyncio
import json
from typing import List

from dotenv import dotenv_values
from google import genai
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from datasets import load_dataset
from src.utils.generate_prompt import generate_prompt
from src.utils.homographs import homographs

# TODO: Use argparse

OUTPUT_FILENAME = "results_gemini_3.jsonl"
MODEL_NAME = "gemini-3-flash-preview"
CONCURRENCY_LIMIT = 80


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    answers: List[str]


async def process_prompt(
    index, prompt, client, MODEL_NAME, semaphore, output_file, file_lock
):
    """Sends prompts to Gemini concurrently and saves to a file."""

    result_entry = {"index": index, "success": False, "content": None, "error": None}

    if not prompt:
        # This will happen if there were no ambiguous words.
        result_entry["error"] = "ERR_EMPTY_PROMPT"
    else:
        # Create a semaphore to limit concurrency
        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": Response.model_json_schema(),
                    },
                )

                output = Response.model_validate_json(response.text)
                result_entry["success"] = True
                result_entry["content"] = output.model_dump()

            except Exception as e:
                result_entry["error"] = str(e)

    async with file_lock:
        # Write to file asynchronously
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    return result_entry


async def main():
    # Set up Gemini API
    config = dotenv_values(".env")
    key = config.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    print("Loading Tatoeba dataset...")
    dataset = load_dataset("tatoeba", "en-tl", lang1="en", lang2="tl")
    sentences = [item["tl"] for item in dataset["train"]["translation"]]

    # NOTE: Can test with a smaller batch first if needed
    # sentences = sentences[:100]

    print(f"{len(sentences)} loaded from dataset. Generating prompts...")
    prompts = []
    for s in sentences:
        words, choices, _ = homographs(s)
        if words:
            prompts.append(
                generate_prompt(sentence=s, pronunciations=choices, words=words)
            )
        else:
            prompts.append(None)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    # Delete existing file if it exists...
    open(OUTPUT_FILENAME, "w").close()

    # Set concurrency limits
    print(f"Concurrency limit: {CONCURRENCY_LIMIT}")
    tasks = [
        asyncio.create_task(
            process_prompt(
                i, prompt, client, MODEL_NAME, semaphore, OUTPUT_FILENAME, file_lock
            )
        )
        for i, prompt in enumerate(prompts, start=1)
    ]

    # Process all tasks while showing an async progress bar
    results = [await t for t in tqdm.as_completed(tasks, total=len(tasks))]

    print("=" * 40)
    print(f"Saved results to {OUTPUT_FILENAME}.")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
