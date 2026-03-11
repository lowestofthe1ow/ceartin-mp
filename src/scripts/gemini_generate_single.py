"""
This script is used to test the Gemini API with a single "generate a sentence
for each possible pronunciation" request.
"""

from typing import List

from dotenv import dotenv_values
from google import genai
from pydantic import BaseModel


class Response(BaseModel):
    """JSON schema for Gemini API's structured output"""

    answers: List[str]


# Set up Gemini API
config = dotenv_values(".env")
key = config["GEMINI_API_KEY"]
client = genai.Client(api_key=key)

prompt = """
<instructions>
The Filipino word given below has several possible pronunciations, given in a
list. For each unique pronunciation, generate a sentence that uses the word
following the definition that corresponds to the pronunciation. The objective
is to create a list of sentences (in Filipino, not in IPA) that use the word
differently, one for each pronunciation. Follow standard Filipino grammar.
Exclude all diacritics from your responses.
</instructions>

<word>baka</word>

<pronunciations_list>
ˈbaka
baˈkaʔ
baˈka
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
    config={
        "response_mime_type": "application/json",
        "response_json_schema": Response.model_json_schema(),
    },
)

output = Response.model_validate_json(response.text)

print(output)
