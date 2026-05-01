import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import re
import hashlib

import numpy as np
import requests


class EmbeddingRetriever:
    def __init__(
        self,
        model_name: str,
        use_hf_api: bool = True,
        hf_api_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_hf_api = use_hf_api
        self.hf_api_token = hf_api_token or os.environ.get("HF_API_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self.model_name.startswith("local-"):
            return self._local_hash_embeddings(texts)
        if self.openai_api_key and not self.use_hf_api:
            return self._openai_embeddings(texts)
        if self.hf_api_token:
            return self._huggingface_embeddings(texts)
        return self._local_hash_embeddings(texts)

    def _local_hash_embeddings(self, texts: List[str], dim: int = 512) -> np.ndarray:
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9$%]+", text.lower())
            for token in tokens:
                idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % dim
                embeddings[row, idx] += 1.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    def _huggingface_embeddings(self, texts: List[str]) -> np.ndarray:
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.hf_api_token}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        output = response.json()
        if isinstance(output, dict) and output.get("error"):
            raise RuntimeError(output["error"])
        if isinstance(output, list):
            return np.asarray(output, dtype=np.float32)
        raise RuntimeError("Unexpected Hugging Face embedding response format")

    def _openai_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is required for OpenAI embeddings. Install it with `pip install openai`.") from exc

        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(model=self.model_name, input=texts)
        return np.asarray([item.embedding for item in response.data], dtype=np.float32)

    def build_index(self, documents: List[Dict[str, str]], index_path: Path, metadata_path: Path) -> np.ndarray:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        texts = [doc["text"] for doc in documents]
        embeddings = self.embed_texts(texts)
        np.save(index_path, embeddings)
        self._save_metadata([doc["metadata"] for doc in documents], metadata_path)
        return embeddings

    def search(self, query: str, index: np.ndarray, top_k: int = 5, return_scores: bool = False):
        query_embedding = self.embed_texts([query])[0]
        if index.ndim == 1:
            index = index.reshape(1, -1)

        query_norm = np.linalg.norm(query_embedding)
        index_norm = np.linalg.norm(index, axis=1)
        similarities = (index @ query_embedding) / (index_norm * query_norm + 1e-10)
        top_indices = np.argsort(-similarities)[:top_k]
        if return_scores:
            return [(int(idx), float(similarities[idx])) for idx in top_indices]
        return top_indices.tolist()

    @staticmethod
    def _save_metadata(metadata: Iterable[Dict[str, str]], output_path: Path) -> None:
        output_path.write_text(json.dumps(list(metadata), indent=2, ensure_ascii=False), encoding="utf-8")


def load_index(index_path: Path) -> np.ndarray:
    return np.load(index_path)
