"""
This script is used to test the Gemini API with a single "generate a sentence
for each possible pronunciation" request.
"""

from typing import List

from dotenv import dotenv_values
from google import genai
from google.genai import types
from pydantic import BaseModel


class Sentence(BaseModel):
    meaning: int
    sentence: str


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    word: str
    answers: List[Sentence]


# Set up Gemini API
config = dotenv_values(".env")
key = config["VERTEX_API_KEY"]
client = genai.Client(api_key=key, vertexai=True)

prompt = """
<instructions>
The Filipino word given below has several possible pronunciations, given in a
list. For each unique pronunciation, generate a sentence that uses the word
following the definition that corresponds to the pronunciation. The objective
is to create a list of sentences (in Filipino, not in IPA) that use the same
word in different meanings or contexts, three for each definition corresponding
to the pronunciations listed. Follow standard Filipino grammar. Exclude all
diacritics from your responses. You may perform conjugations if applicable.
Include the word itself in your response, then index the sentence's use of the
word starting at meaning=1.
</instructions>
<word>sikat</word>
<pronunciations_list>
si'kat
'sikat
</pronunciations_list>
"""

print("=" * 40)
print("The following prompt will be fed to Gemini:")
print("-" * 40)
print(prompt)
print("=" * 40)

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=Response.model_json_schema(),
        # This is where thinking_level lives
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL"  # Use uppercase or "minimal"
        ),
    ),
)

output = Response.model_validate_json(response.text)
print(output)
