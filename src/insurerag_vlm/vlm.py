import json
import os
from typing import Dict, Optional

import requests


class VLMClient:
    def __init__(self, model_name: str, hf_api_token: Optional[str] = None, openai_api_key: Optional[str] = None, use_hf_api: bool = True):
        self.model_name = model_name
        self.hf_api_token = hf_api_token or os.environ.get("HF_API_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.use_hf_api = use_hf_api

    def generate(self, prompt: str) -> str:
        if self.use_hf_api:
            return self._call_huggingface(prompt)
        return self._call_openai(prompt)

    def _call_huggingface(self, prompt: str) -> str:
        if not self.hf_api_token:
            raise ValueError("HF_API_TOKEN is required for Hugging Face inference.")

        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.hf_api_token}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        output = response.json()
        if isinstance(output, dict) and output.get("error"):
            raise RuntimeError(output["error"])
        if isinstance(output, list) and output:
            return output[0].get("generated_text", "").strip()
        return json.dumps(output)

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai is required for OpenAI integration. Install it with `pip install openai`.") from exc

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI calls.")
        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()


def format_prompt(context: str, question: str, template: str) -> str:
    return template.format(context=context.strip(), question=question.strip())
