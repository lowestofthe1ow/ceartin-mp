"""
This script concurrently sends a series of requests to the Gemini API to build
the PhoneticTatoeba dataset.
"""

import argparse
import asyncio

from dotenv import dotenv_values
from google import genai
from tqdm.asyncio import tqdm

from src.utils.generate_prompt.transcribe import generate_prompt
from src.utils.homographs import homographs
from src.utils.process_prompt import process_prompt

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path",
    type=str,
    default="data/newsph-nli/newsph-nli.txt",
)

args = parser.parse_args()

OUTPUT_FILENAME = "results_gemini_2.5_lite.jsonl"
MODEL_NAME = "gemini-2.5-flash-lite"
CONCURRENCY_LIMIT = 80


async def main():
    # Set up Gemini API
    config = dotenv_values(".env")
    key = config.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    # NOTE: Can test with a smaller batch first if needed
    # sentences = sentences[:100]

    print("Here")
    with open(args.path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"{len(sentences)} loaded from dataset. Generating prompts...")
    prompts = []
    for s in tqdm(sentences):
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
