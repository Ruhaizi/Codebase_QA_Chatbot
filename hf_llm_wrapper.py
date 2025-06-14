from langchain_core.language_models import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List
from pydantic import Field, PrivateAttr


class HuggingFaceInferenceLLM(LLM):
    repo_id: str
    token: str
    max_new_tokens: int = 256
    temperature: float = 0.7

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = InferenceClient(model=self.repo_id, token=self.token, timeout=120)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            return_full_text=False
        )
        return response.strip()

    @property
    def _llm_type(self) -> str:
        return "custom_hf_inference"
