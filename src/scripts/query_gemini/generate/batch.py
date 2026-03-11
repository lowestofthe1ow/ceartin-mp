"""
This script is used to test the Gemini API with a single request.
"""

import asyncio
import json
from typing import List

from dotenv import dotenv_values
from google import genai
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from src.datasets.wikipron.wikipron_tl_df import wikipron_tl_df
from src.utils.generate_prompt.generate import generate_prompt
from src.utils.homographs import fill_template, homographs
from src.utils.process_prompt import process_prompt

config = dotenv_values(".env")

# TODO: Use argparse

FILE_PATH = config["WIKIPRON_PATH"]
HOMOGRAPHS, _ = wikipron_tl_df(FILE_PATH)

OUTPUT_FILENAME = "results_gemini_3.jsonl"
MODEL_NAME = "gemini-3-flash-preview"
CONCURRENCY_LIMIT = 80


async def main():
    # Set up Gemini API
    config = dotenv_values(".env")
    key = config.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    prompts = [generate_prompt(word) for word in HOMOGRAPHS.keys()]

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
