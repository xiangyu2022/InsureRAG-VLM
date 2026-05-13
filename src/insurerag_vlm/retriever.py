import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import requests


TOKEN_RE = re.compile(r"[a-zA-Z0-9$%]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


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
        self._local_tokenizer = None
        self._local_model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        if self.model_name.startswith("local-"):
            return self._local_hash_embeddings(texts)
        if Path(self.model_name).exists():
            return self._local_transformer_embeddings(texts)
        if self.openai_api_key and not self.use_hf_api:
            return self._openai_embeddings(texts)
        if self.hf_api_token:
            return self._huggingface_embeddings(texts)
        return self._local_hash_embeddings(texts)

    def _local_hash_embeddings(self, texts: List[str], dim: int = 512) -> np.ndarray:
        embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
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

    def _ensure_local_transformer(self):
        if self._local_model is not None and self._local_tokenizer is not None:
            return self._local_tokenizer, self._local_model
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Local transformer embeddings require torch and transformers.") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModel.from_pretrained(self.model_name)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        self._local_tokenizer = tokenizer
        self._local_model = model
        return tokenizer, model

    def _local_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        import torch

        tokenizer, model = self._ensure_local_transformer()
        device = next(model.parameters()).device
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(texts), 16):
                batch = texts[start : start + 16]
                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                result = model(**encoded)
                hidden = result.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs)

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


class SparseRetriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def build_index(self, texts: List[str], index_path: Path) -> Dict[str, object]:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        doc_term_freqs: List[Dict[str, int]] = []
        doc_lengths: List[int] = []
        document_frequency: Counter[str] = Counter()

        for text in texts:
            tokens = tokenize(text)
            counts = Counter(tokens)
            doc_term_freqs.append(dict(counts))
            doc_lengths.append(len(tokens))
            document_frequency.update(counts.keys())

        avgdl = sum(doc_lengths) / max(1, len(doc_lengths))
        doc_count = len(doc_term_freqs)
        idf = {
            term: math.log(1 + ((doc_count - df + 0.5) / (df + 0.5)))
            for term, df in document_frequency.items()
        }
        payload = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": avgdl,
            "doc_lengths": doc_lengths,
            "idf": idf,
            "doc_term_freqs": doc_term_freqs,
        }
        index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def search(
        self,
        query: str,
        index: Dict[str, object],
        top_k: int = 5,
        return_scores: bool = False,
    ):
        query_terms = tokenize(query)
        if not query_terms:
            return [] if not return_scores else []

        k1 = float(index.get("k1", self.k1))
        b = float(index.get("b", self.b))
        avgdl = float(index.get("avgdl", 0.0))
        doc_lengths = index.get("doc_lengths", [])
        idf = index.get("idf", {})
        doc_term_freqs = index.get("doc_term_freqs", [])
        scores = np.zeros(len(doc_term_freqs), dtype=np.float32)

        for doc_idx, term_freqs in enumerate(doc_term_freqs):
            doc_len = float(doc_lengths[doc_idx]) if doc_idx < len(doc_lengths) else 0.0
            denom_norm = k1 * (1 - b + b * (doc_len / max(avgdl, 1e-10)))
            score = 0.0
            for term in query_terms:
                freq = float(term_freqs.get(term, 0.0))
                if freq <= 0.0:
                    continue
                term_idf = float(idf.get(term, 0.0))
                numerator = freq * (k1 + 1.0)
                denominator = freq + denom_norm
                score += term_idf * (numerator / max(denominator, 1e-10))
            scores[doc_idx] = score

        top_indices = np.argsort(-scores)[:top_k]
        if return_scores:
            return [(int(idx), float(scores[idx])) for idx in top_indices if float(scores[idx]) > 0.0]
        return [int(idx) for idx in top_indices if float(scores[idx]) > 0.0]


def load_index(index_path: Path) -> np.ndarray:
    return np.load(index_path)


def load_sparse_index(index_path: Path) -> Dict[str, object]:
    return json.loads(Path(index_path).read_text(encoding="utf-8"))
