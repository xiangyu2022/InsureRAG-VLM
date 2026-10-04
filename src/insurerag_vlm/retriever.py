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


class EmbeddingBackendError(RuntimeError):
    """The requested embedding backend cannot run; no substitute was used."""


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


class EmbeddingRetriever:
    def __init__(
        self,
        model_name: str,
        use_hf_api: bool = True,
        hf_api_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        pooling: str = "auto",
        query_instruction: str = "",
        max_length: Optional[int] = None,
    ):
        self.model_name = str(model_name).strip()
        self.use_hf_api = use_hf_api
        self.hf_api_token = hf_api_token or os.environ.get("HF_API_TOKEN")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._local_tokenizer = None
        self._local_model = None
        if pooling not in {"auto", "cls", "mean"}:
            raise EmbeddingBackendError("Embedding pooling must be auto, cls, or mean.")
        if max_length is not None and max_length < 1:
            raise EmbeddingBackendError("Embedding max_length must be positive.")
        self.pooling = pooling
        self.query_instruction = query_instruction
        self.max_length = max_length
        self._fingerprint_cache_key = None
        self._fingerprint_cache = None
        self._loaded_checkpoint_sha256 = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        backend = self._resolve_backend()
        if backend == "local-hashing":
            return self._local_hash_embeddings(texts)
        if backend == "local-transformer":
            return self._local_transformer_embeddings(texts)
        if backend == "openai":
            return self._openai_embeddings(texts)
        if backend == "huggingface":
            return self._huggingface_embeddings(texts)
        raise EmbeddingBackendError(f"Unsupported embedding backend: {backend}")

    def _resolve_backend(self) -> str:
        if self.model_name == "local-hashing":
            return "local-hashing"
        model_path = Path(self.model_name)
        if model_path.is_dir():
            if not (model_path / "config.json").is_file():
                raise EmbeddingBackendError(
                    f"Local embedding checkpoint {self.model_name!r} has no config.json; no hashing fallback was used."
                )
            return "local-transformer"
        local_path_requested = (
            model_path.is_absolute()
            or self.model_name.startswith(("./", "../", "models/", "models\\"))
            or "\\" in self.model_name
        )
        if local_path_requested:
            raise EmbeddingBackendError(f"Local embedding checkpoint is unavailable: {self.model_name!r}.")
        if not self.model_name or self.model_name.startswith("local-"):
            raise EmbeddingBackendError("Select the explicit local-hashing baseline or an available embedding model.")
        if not self.use_hf_api:
            if not self.openai_api_key:
                raise EmbeddingBackendError("OpenAI embeddings were selected but OPENAI_API_KEY is unavailable.")
            return "openai"
        if not self.hf_api_token:
            raise EmbeddingBackendError(
                f"Embedding model {self.model_name!r} is not a local checkpoint and HF_API_TOKEN is unavailable. "
                "Select local-hashing explicitly to run the CPU baseline."
            )
        return "huggingface"

    @staticmethod
    def _read_config(path: Path) -> Dict[str, object]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EmbeddingBackendError(f"Cannot read embedding configuration: {path}") from exc
        if not isinstance(payload, dict):
            raise EmbeddingBackendError(f"Embedding configuration must be a JSON object: {path}")
        return payload

    def _local_encoding_config(self) -> Dict[str, object]:
        root = Path(self.model_name)
        model_config = self._read_config(root / "config.json")
        sidecar_path = root / "insurerag_embedding_config.json"
        sidecar = self._read_config(sidecar_path) if sidecar_path.exists() else {}
        pooling_path = root / "1_Pooling" / "config.json"
        modules_path = root / "modules.json"
        if modules_path.exists():
            try:
                modules = json.loads(modules_path.read_text(encoding="utf-8"))
                pooling_modules = [module for module in modules if str(module.get("type", "")).endswith(".Pooling")]
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                raise EmbeddingBackendError(f"Invalid sentence-transformer module manifest: {modules_path}") from exc
            supported_modules = {"sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling", "sentence_transformers.models.Normalize"}
            if any(module.get("type") not in supported_modules for module in modules):
                raise EmbeddingBackendError("Checkpoint contains unsupported sentence-transformer modules; they cannot be silently skipped.")
            if len(pooling_modules) > 1:
                raise EmbeddingBackendError("Multiple pooling modules are unsupported; supply a single-pooling checkpoint.")
            if pooling_modules:
                pooling_path = root / str(pooling_modules[0].get("path", "")) / "config.json"
                if not pooling_path.resolve().is_relative_to(root.resolve()):
                    raise EmbeddingBackendError("Pooling configuration must remain inside the checkpoint directory.")
        pooling, pooling_source = self.pooling, "explicit_argument"
        if pooling == "auto" and sidecar.get("pooling"):
            pooling, pooling_source = str(sidecar["pooling"]), sidecar_path.name
        if pooling == "auto" and pooling_path.exists():
            config = self._read_config(pooling_path)
            enabled = {key for key, value in config.items() if key.startswith("pooling_mode_") and value is True}
            if enabled == {"pooling_mode_cls_token"}:
                pooling = "cls"
            elif enabled == {"pooling_mode_mean_tokens"}:
                pooling = "mean"
            else:
                raise EmbeddingBackendError(f"Unsupported or ambiguous pooling configuration: {pooling_path}")
            pooling_source = pooling_path.relative_to(root).as_posix()
        if pooling not in {"cls", "mean"}:
            raise EmbeddingBackendError(
                "Local checkpoint does not declare supported pooling. Set retrieval_pooling to cls or mean, "
                "or provide insurerag_embedding_config.json with a pooling field."
            )
        if sidecar.get("normalize", True) is not True:
            raise EmbeddingBackendError("This retriever requires L2-normalized embeddings; checkpoint declares otherwise.")
        sentence_config_path = root / "sentence_bert_config.json"
        sentence_config = self._read_config(sentence_config_path) if sentence_config_path.exists() else {}
        max_length = self.max_length or sidecar.get("max_length") or sentence_config.get("max_seq_length") or model_config.get("max_position_embeddings")
        if not isinstance(max_length, int) or max_length < 1:
            raise EmbeddingBackendError("Checkpoint has no valid maximum sequence length; set retrieval_max_length explicitly.")
        model_limit = model_config.get("max_position_embeddings")
        if isinstance(model_limit, int) and max_length > model_limit:
            raise EmbeddingBackendError(f"Embedding max_length {max_length} exceeds checkpoint position limit {model_limit}.")
        return {
            "pooling": pooling, "pooling_source": pooling_source,
            "max_length": max_length, "query_instruction": self.query_instruction,
            "normalization": "l2", "embedding_dim": model_config.get("hidden_size", model_config.get("d_model")),
        }

    def index_fingerprint(self) -> Dict[str, object]:
        """Describe the encoding contract without credentials or transient timing.

        Local content hashes are cached until relevant file sizes/mtimes change.
        This is checked by the index loader before reusing stored dense vectors.
        """
        backend = self._resolve_backend()
        payload = {
            "schema_version": 1, "backend": backend, "model": self.model_name,
            "is_baseline": backend == "local-hashing", "query_instruction": self.query_instruction,
            "pooling": None, "max_length": None, "normalization": "provider-defined",
            "embedding_dim": None,
        }
        if backend == "local-hashing":
            return {**payload, "hash_algorithm": "sha256-token-count-v1", "embedding_dim": 512, "normalization": "l2"}
        if backend != "local-transformer":
            return payload
        root = Path(self.model_name).resolve()
        candidates = list(root.iterdir()) + list(root.glob("*/config.json"))
        relevant = sorted(path for path in candidates if path.is_file() and (
            path.suffix in {".json", ".safetensors", ".bin", ".model"}
            or path.name in {"vocab.txt", "merges.txt"}
        ) and ".cache" not in path.relative_to(root).parts)
        key = (str(root), self.pooling, self.query_instruction, self.max_length,
               tuple((path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns) for path in relevant))
        if key == self._fingerprint_cache_key:
            return json.loads(json.dumps(self._fingerprint_cache))
        files = {}
        for path in relevant:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            files[path.relative_to(root).as_posix()] = digest.hexdigest()
        fingerprint = {
            **payload, "model": str(root), **self._local_encoding_config(),
            "checkpoint_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest(),
            "checkpoint_files_sha256": files,
        }
        self._fingerprint_cache_key, self._fingerprint_cache = key, fingerprint
        return json.loads(json.dumps(fingerprint))

    def backend_metadata(self) -> Dict[str, object]:
        return self.index_fingerprint()

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
        checkpoint_sha256 = self.index_fingerprint()["checkpoint_sha256"]
        if self._local_model is not None and self._local_tokenizer is not None:
            if self._loaded_checkpoint_sha256 == checkpoint_sha256:
                return self._local_tokenizer, self._local_model
            self._local_tokenizer, self._local_model = None, None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Local transformer embeddings require torch and transformers.") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
        self._local_tokenizer = tokenizer
        self._local_model = model
        self._loaded_checkpoint_sha256 = checkpoint_sha256
        return tokenizer, model

    def _local_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        import torch

        encoding_config = self._local_encoding_config()
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
                    max_length=encoding_config["max_length"],
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                result = model(**encoded)
                hidden = result.last_hidden_state
                if encoding_config["pooling"] == "cls":
                    pooled = hidden[:, 0]
                else:
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
        index_path.with_suffix(".manifest.json").write_text(
            json.dumps({"embedding_fingerprint": self.index_fingerprint()}, indent=2), encoding="utf-8"
        )
        return embeddings

    def search(self, query: str, index: np.ndarray, top_k: int = 5, return_scores: bool = False):
        if index.size == 0 or top_k <= 0:
            return []
        query_embedding = self.embed_texts([self.query_instruction + query])[0]
        if index.ndim == 1:
            index = index.reshape(1, -1)
        if index.ndim != 2 or index.shape[1] != query_embedding.shape[0]:
            raise EmbeddingBackendError("Dense index dimensions do not match the requested embedding backend; rebuild the index.")

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
