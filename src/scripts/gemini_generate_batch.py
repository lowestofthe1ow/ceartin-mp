"""
This script concurrently sends "generate a sentence for each possible
pronunciation" requests to the Gemini API via Vertex AI.
"""

import asyncio

from dotenv import dotenv_values
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

from src.datasets.wikipron_tl_df import wikipron_tl_df
from src.utils.generate_prompt.synthesize import generate_prompt
from src.utils.process_prompt import process_prompt

# Read from .env
config = dotenv_values(".env")
FILE_PATH = config["WIKIPRON_PATH"]
VERTEX_KEY = config["VERTEX_API_KEY"]

HOMOGRAPHS, _ = wikipron_tl_df(FILE_PATH)
CONCURRENCY_LIMIT = 80
OUTPUT_FILENAME = "results_gemini_3.jsonl"
MODEL_NAME = "gemini-3-flash-preview"


async def main():
    # Set up Gemini API for Vertex AI
    client = genai.Client(api_key=VERTEX_KEY, vertexai=True)

    prompts = [generate_prompt(word) for word in HOMOGRAPHS.keys()]

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()

    # Delete existing file if it exists...
    open(OUTPUT_FILENAME, "w").close()

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Response.model_json_schema(),
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )

    # Set concurrency limits
    print(f"Concurrency limit: {CONCURRENCY_LIMIT}")

    tasks = [
        asyncio.create_task(
            process_prompt(
                i,
                prompt,
                client,
                MODEL_NAME,
                semaphore,
                OUTPUT_FILENAME,
                file_lock,
                gen_config,
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
