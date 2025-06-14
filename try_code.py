from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient(
    model="microsoft/Phi-3-mini-4k-instruct",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    timeout=120
)


def call_llm(client: InferenceClient, prompt: str):
    response = client.text_generation(
        prompt,
        max_new_tokens=200,
        temperature=0.7,
        return_full_text=False  # only return the new content
    )
    return response

# Example call
output = call_llm(client, "Write me a crazy joke")
print(output)
