"""
This script is used to test the Gemini API with a single "fill in the
pronunciation" request.
"""

from typing import List

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.utils.generate_prompt.transcribe import generate_prompt
from src.utils.homographs import fill_template, homographs


# TODO: Use argparse
class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    answers: List[str]


# Set up Gemini API
config = dotenv_values(".env")
VERTEX_KEY = config["VERTEX_API_KEY"]
client = genai.Client(api_key=VERTEX_KEY, vertexai=True)

gen_config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_json_schema=Response.model_json_schema(),
    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
)


# TODO: This is a sample sentence
SENTENCE = "Hindi ko ugali ang mamulitika; mas gusto kong tahimik na magtrabaho. Pero sasabihin ko ito ngayon: ang tapang, lakas, at diskarte, hindi nadadaan sa mapanirang salita. Ang kailangan ng taumbayan ay tapang sa gawa, ayon kay Robredo sa inilabas nitong statement."

# "Ang sikolohiya ay mahalaga sa edukasyon dahil ito'y nag-aaral ng mga proseso ng pag-aaral at pag-unlad ng isip at damdamin ng mga mag-aaral. Ito'y nagbibigay linaw sa mga guro kung paano turuan ang mga mag-aaral nang mas epektibo batay sa kanilang pangangailangan"

# Extract homograph information given the sentence
ambiguous_words, choices, output_template = homographs(SENTENCE)

prompt = generate_prompt(
    sentence=SENTENCE, pronunciations=choices, words=ambiguous_words
)

print("=" * 40)
print("The following prompt will be fed to Gemini:")
print("-" * 40)
print(prompt)
print("=" * 40)

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents=prompt, config=gen_config
)

output = Response.model_validate_json(response.text)

print(fill_template(output_template, output.answers))
