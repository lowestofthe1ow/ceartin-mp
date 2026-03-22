import json
from typing import List

from pydantic import BaseModel


class Sentence(BaseModel):
    pronunciation: str
    sentence: str


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    word: str
    answers: List[Sentence]


async def process_prompt(
    index, prompt, client, model_name, semaphore, output_file, file_lock, gen_config
):
    """
    Sends prompts to Gemini concurrently and saves to an entry in an output
    .jsonl file.
    """

    result_entry = {"index": index, "success": False, "content": None, "error": None}

    if not prompt:
        # This will happen if there were no ambiguous words.
        result_entry["error"] = "ERR_EMPTY_PROMPT"
    else:
        # Create a semaphore to limit concurrency
        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name, contents=prompt, config=gen_config
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
