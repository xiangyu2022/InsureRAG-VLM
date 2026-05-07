import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .retriever import EmbeddingRetriever

SUPPORTED_VISUAL_BACKENDS = {
    "visual_stub",
    "local_image",
    "colqwen2_hf",
    "colqwen2_local",
    "colpali_hf",
    "colpali_local",
}
TEXT_EMBED_DIM = 512
IMAGE_FEATURE_DIM = 32
LOCAL_IMAGE_TEXT_WEIGHT = 0.98
LOCAL_IMAGE_LAYOUT_WEIGHT = 0.02
HF_VISUAL_BACKENDS = {"colqwen2_hf", "colqwen2_local", "colpali_hf", "colpali_local"}


@dataclass
class VisualIndexResult:
    backend: str
    index_path: Path
    metadata_path: Path
    page_count: int


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _visual_stub_text(page: Dict[str, Any]) -> str:
    parts = [
        page.get("section_hint") or "",
        page.get("text_layer") or "",
        page.get("ocr_text") or "",
        page.get("source") or "",
        page.get("doc_id") or "",
        page.get("version_id") or "",
        str(page.get("page_number") or page.get("page_index") or ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _candidate_ocr_aux_paths(page_manifest_path: Path) -> List[Path]:
    path = Path(page_manifest_path)
    return [
        path.parent / "ocr_aux.jsonl",
        path.parent.parent / "02_processed" / "ocr_aux.jsonl",
        path.parent.parent.parent / "02_processed" / "ocr_aux.jsonl",
        Path("data/02_processed/ocr_aux.jsonl"),
    ]


def _enrich_pages_with_aux_text(pages: List[Dict[str, Any]], page_manifest_path: Path) -> List[Dict[str, Any]]:
    aux_path = next((candidate for candidate in _candidate_ocr_aux_paths(page_manifest_path) if candidate.exists()), None)
    if aux_path is None:
        return pages

    aux_by_page = {item.get("page_id"): item for item in _read_jsonl(aux_path)}
    enriched_pages = []
    for page in pages:
        enriched = dict(page)
        aux = aux_by_page.get(page.get("page_id"), {})
        if aux:
            enriched["text_layer"] = aux.get("text_layer", "")
            enriched["ocr_text"] = aux.get("ocr_text", "")
            enriched["aux_text_chars"] = len(enriched["text_layer"] or enriched["ocr_text"] or "")
        enriched_pages.append(enriched)
    return enriched_pages


def _resolve_image_path(image_path: str | None, manifest_path: Path | None = None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.exists():
        return path
    if manifest_path is not None:
        candidate = manifest_path.parent / path
        if candidate.exists():
            return candidate
    return path


def _image_feature_vector(image_path: str | None, manifest_path: Path | None = None) -> np.ndarray:
    """Return compact page-layout features for a local, no-GPU visual baseline."""
    features = np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
    path = _resolve_image_path(image_path, manifest_path)
    if path is None or not path.exists():
        return features

    try:
        from PIL import Image, ImageFilter
    except ImportError:
        return features

    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            width, height = gray.size
            small = gray.resize((64, 64))
            arr = np.asarray(small, dtype=np.float32) / 255.0
            ink = (arr < 0.92).astype(np.float32)
            edges = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0

            hist = np.histogram(arr, bins=12, range=(0.0, 1.0), density=False)[0].astype(np.float32)
            hist = hist / (hist.sum() + 1e-10)
            row_density = ink.mean(axis=1)
            col_density = ink.mean(axis=0)
            row_bins = np.array_split(row_density, 6)
            col_bins = np.array_split(col_density, 6)

            values = [
                float(width / max(1, height)),
                float(width),
                float(height),
                float(ink.mean()),
                float(ink.std()),
                float(edges.mean()),
                float(edges.std()),
                float(ink[:16, :].mean()),
            ]
            values.extend(float(x.mean()) for x in row_bins)
            values.extend(float(x.mean()) for x in col_bins)
            values.extend(float(x) for x in hist)

            vector = np.asarray(values[:IMAGE_FEATURE_DIM], dtype=np.float32)
            features[: len(vector)] = vector
            norm = np.linalg.norm(features)
            return features / (norm + 1e-10)
    except Exception:
        return features


def _query_layout_hint_vector(query: str) -> np.ndarray:
    """Map policy-document query intents to a tiny visual prior vector."""
    if os.environ.get("INSURERAG_VISUAL_QUERY_HINTS") != "1":
        return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
    lowered = query.lower()
    hints = np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)
    if any(term in lowered for term in ["declaration", "declarations", "limit", "premium"]):
        hints[7] = 0.8  # top-page/table-heavy signal
        hints[3] = 0.4
    if any(term in lowered for term in ["deductible", "coverage", "liability"]):
        hints[3] = max(hints[3], 0.5)
        hints[5] = 0.25
    if any(term in lowered for term in ["endorsement", "rider", "exclusion", "duties"]):
        hints[4] = 0.4
        hints[5] = 0.3
    norm = np.linalg.norm(hints)
    return hints / (norm + 1e-10) if norm else hints


def _page_text_embeddings(texts: List[str]) -> np.ndarray:
    retriever = EmbeddingRetriever("local-hashing")
    embeddings = retriever.embed_texts(texts)
    if embeddings.shape[1] != TEXT_EMBED_DIM:
        padded = np.zeros((embeddings.shape[0], TEXT_EMBED_DIM), dtype=np.float32)
        width = min(TEXT_EMBED_DIM, embeddings.shape[1])
        padded[:, :width] = embeddings[:, :width]
        return padded
    return embeddings


def _hf_backend_family(backend: str) -> str:
    if backend.startswith("colqwen2"):
        return "colqwen2"
    if backend.startswith("colpali"):
        return "colpali"
    raise ValueError(f"Unsupported Hugging Face visual backend: {backend}")


def _hf_model_name(backend: str) -> str:
    family = _hf_backend_family(backend)
    if family == "colqwen2":
        return os.environ.get("INSURERAG_COLQWEN2_MODEL", "vidore/colqwen2-v1.0-hf")
    return os.environ.get("INSURERAG_COLPALI_MODEL", "vidore/colpali-v1.3-hf")


def _hf_index_path(index_dir: Path, backend: str) -> Path:
    return Path(index_dir) / f"{backend}.pt"


def _embedding_dir(index_dir: Path, backend: str) -> Path:
    return Path(index_dir) / "page_embeddings" / backend


def _import_torch_and_transformers(backend: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for ColPali/ColQwen2 backends.") from exc

    try:
        if _hf_backend_family(backend) == "colqwen2":
            from transformers import ColQwen2ForRetrieval, ColQwen2Processor
            from transformers.utils.import_utils import is_flash_attn_2_available

            return torch, ColQwen2ForRetrieval, ColQwen2Processor, is_flash_attn_2_available
        from transformers import ColPaliForRetrieval, ColPaliProcessor

        return torch, ColPaliForRetrieval, ColPaliProcessor, None
    except ImportError as exc:
        raise ImportError(
            "A recent transformers version with ColPali/ColQwen2 retrieval classes is required. "
            "Install/upgrade with: pip install -U 'transformers>=4.53' accelerate safetensors"
        ) from exc


def _load_hf_model_and_processor(backend: str):
    torch, model_cls, processor_cls, is_flash_attn_2_available = _import_torch_and_transformers(backend)
    if os.environ.get("INSURERAG_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
        raise RuntimeError("INSURERAG_REQUIRE_CUDA=1 but CUDA is not available.")

    model_name = _hf_model_name(backend)
    dtype_name = os.environ.get("INSURERAG_VISUAL_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    device_map = os.environ.get("INSURERAG_VISUAL_DEVICE_MAP", "auto")
    kwargs: Dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
    }
    if _hf_backend_family(backend) == "colqwen2" and is_flash_attn_2_available is not None:
        kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    try:
        model = model_cls.from_pretrained(model_name, **kwargs).eval()
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(model_name, **kwargs).eval()
    processor = processor_cls.from_pretrained(model_name)
    return torch, model, processor, model_name


def _model_device(model: Any):
    if hasattr(model, "device"):
        return model.device
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def _load_page_images(pages: List[Dict[str, Any]], manifest_path: Path | None = None) -> List[Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to load page images for visual retrieval.") from exc

    images = []
    for page in pages:
        image_path = _resolve_image_path(page.get("image_path"), manifest_path)
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Missing page image for page_id={page.get('page_id')}: {page.get('image_path')}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _batch_items(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _build_hf_visual_index(page_manifest_path: Path, index_dir: Path, backend: str) -> VisualIndexResult:
    torch, model, processor, model_name = _load_hf_model_and_processor(backend)
    pages = _enrich_pages_with_aux_text(_read_jsonl(page_manifest_path), page_manifest_path)
    batch_size = int(os.environ.get("INSURERAG_VISUAL_BATCH_SIZE", "2"))
    device = _model_device(model)
    embedding_dir = _embedding_dir(index_dir, backend)
    embedding_dir.mkdir(parents=True, exist_ok=True)

    embeddings = []
    with torch.no_grad():
        for batch_pages in _batch_items(pages, batch_size):
            images = _load_page_images(batch_pages, page_manifest_path)
            inputs = processor(images=images, return_tensors="pt").to(device)
            batch_embeddings = model(**inputs).embeddings
            for embedding in batch_embeddings:
                embeddings.append(embedding.detach().cpu())

    metadata_path = Path(index_dir) / f"{backend}_pages.jsonl"
    index_path = _hf_index_path(index_dir, backend)
    _write_jsonl(pages, metadata_path)
    torch.save(
        {
            "backend": backend,
            "family": _hf_backend_family(backend),
            "model_name": model_name,
            "embeddings": embeddings,
            "metadata_path": str(metadata_path),
        },
        index_path,
    )
    return VisualIndexResult(backend, index_path, metadata_path, len(pages))


def _hf_visual_search(query: str, index_dir: Path, backend: str, top_k: int) -> List[Dict[str, Any]]:
    torch, model, processor, model_name = _load_hf_model_and_processor(backend)
    index_path = _hf_index_path(index_dir, backend)
    metadata_path = Path(index_dir) / f"{backend}_pages.jsonl"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Visual index not found for backend={backend} in {index_dir}")

    try:
        payload = torch.load(index_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(index_path, map_location="cpu")
    if payload.get("model_name") and payload["model_name"] != model_name:
        raise RuntimeError(
            f"Index was built with {payload['model_name']}, but current backend loads {model_name}. "
            "Rebuild the visual index or set the matching INSURERAG_*_MODEL env var."
        )
    page_embeddings = payload["embeddings"]
    pages = _read_jsonl(metadata_path)
    device = _model_device(model)

    with torch.no_grad():
        inputs = processor(text=[query], return_tensors="pt").to(device)
        query_embeddings = model(**inputs).embeddings.detach().cpu()
        scores = processor.score_retrieval(
            query_embeddings,
            page_embeddings,
            batch_size=int(os.environ.get("INSURERAG_SCORE_BATCH_SIZE", "128")),
            output_device="cpu",
        )[0]

    top_indices = torch.argsort(scores, descending=True)[:top_k].tolist()
    ranked_pages = []
    for idx in top_indices:
        page = pages[int(idx)]
        ranked_pages.append(
            {
                "page_id": page.get("page_id"),
                "image_path": page.get("image_path"),
                "score": float(scores[int(idx)]),
                "source": page.get("source") or page.get("page_id"),
                "page_number": page.get("page_number"),
                "section_hint": page.get("section_hint"),
                "backend": backend,
                "model_name": model_name,
            }
        )
    return ranked_pages


def build_visual_index(
    page_manifest_path: Path,
    index_dir: Path,
    backend: str = "visual_stub",
) -> VisualIndexResult:
    if backend not in SUPPORTED_VISUAL_BACKENDS:
        raise ValueError(f"Unsupported visual backend: {backend}. Available: {sorted(SUPPORTED_VISUAL_BACKENDS)}")

    page_manifest_path = Path(page_manifest_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    if backend in HF_VISUAL_BACKENDS:
        return _build_hf_visual_index(page_manifest_path, index_dir, backend)

    pages = _enrich_pages_with_aux_text(_read_jsonl(page_manifest_path), page_manifest_path)
    texts = [_visual_stub_text(page) for page in pages]
    text_embeddings = _page_text_embeddings(texts)

    if backend == "local_image":
        image_features = np.vstack(
            [_image_feature_vector(page.get("image_path"), page_manifest_path) for page in pages]
        ).astype(np.float32) if pages else np.zeros((0, IMAGE_FEATURE_DIM), dtype=np.float32)
        embeddings = np.hstack([LOCAL_IMAGE_TEXT_WEIGHT * text_embeddings, LOCAL_IMAGE_LAYOUT_WEIGHT * image_features]).astype(np.float32)
        enriched_pages = []
        for page, features in zip(pages, image_features):
            enriched = dict(page)
            enriched["image_stats"] = {
                "layout_density": float(features[3]) if len(features) > 3 else 0.0,
                "edge_density": float(features[5]) if len(features) > 5 else 0.0,
                "feature_dim": IMAGE_FEATURE_DIM,
            }
            enriched_pages.append(enriched)
        pages = enriched_pages
    else:
        embeddings = text_embeddings

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-10)

    index_path = index_dir / f"{backend}.npy"
    metadata_path = index_dir / f"{backend}_pages.jsonl"
    np.save(index_path, embeddings)
    _write_jsonl(pages, metadata_path)
    return VisualIndexResult(backend, index_path, metadata_path, len(pages))


def visual_search(
    query: str,
    index_dir: Path,
    backend: str = "visual_stub",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if backend in HF_VISUAL_BACKENDS:
        return _hf_visual_search(query, Path(index_dir), backend, top_k)

    index_path = Path(index_dir) / f"{backend}.npy"
    metadata_path = Path(index_dir) / f"{backend}_pages.jsonl"
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Visual index not found for backend={backend} in {index_dir}")

    index = np.load(index_path)
    pages = _read_jsonl(metadata_path)
    query_text = _page_text_embeddings([query])[0]
    if backend == "local_image":
        query_embedding = np.concatenate(
            [LOCAL_IMAGE_TEXT_WEIGHT * query_text, LOCAL_IMAGE_LAYOUT_WEIGHT * _query_layout_hint_vector(query)]
        ).astype(np.float32)
    else:
        query_embedding = query_text
    similarities = (index @ query_embedding) / (
        np.linalg.norm(index, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    candidate_pool = min(len(similarities), max(top_k, top_k * 4, 20))
    top_indices = np.argsort(-similarities)[:candidate_pool]

    ranked_pages = []
    for idx in top_indices:
        page = pages[int(idx)]
        source = page.get("source") or page.get("page_id")
        rerank_score = 0.0
        if backend in {"visual_stub", "local_image"}:
            from .pipeline import DocumentRetrievalPipeline

            rerank_score = DocumentRetrievalPipeline._score_insurance_evidence(
                query,
                _visual_stub_text(page),
                source,
                page.get("page_number"),
            )
        ranked_pages.append(
            {
                "page_id": page.get("page_id"),
                "image_path": page.get("image_path"),
                "score": round(float(similarities[idx]) + rerank_score, 6),
                "retrieval_score": float(similarities[idx]),
                "rerank_score": round(rerank_score, 6),
                "source": source,
                "page_number": page.get("page_number"),
                "section_hint": page.get("section_hint"),
                "image_stats": page.get("image_stats"),
                "backend": backend,
            }
        )
    ranked_pages.sort(key=lambda page: float(page.get("score", 0.0)), reverse=True)
    return ranked_pages[:top_k]


def _source_page_number(source: str) -> int | None:
    match = re.search(r"#page=(\d+)", source or "")
    return int(match.group(1)) if match else None


def _source_doc_name(source: str) -> str:
    source = (source or "").split("#page=", 1)[0]
    return Path(source).name


def _is_hit(candidate: Dict[str, Any], gold_sources: set[str], gold_page_ids: set[str]) -> bool:
    if candidate.get("source") in gold_sources or candidate.get("page_id") in gold_page_ids:
        return True
    candidate_page = candidate.get("page_number")
    candidate_doc = _source_doc_name(str(candidate.get("source") or ""))
    return any(
        _source_page_number(source) == candidate_page
        and (not candidate_doc or candidate_doc == _source_doc_name(source))
        for source in gold_sources
    )


def compute_visual_retrieval_metrics(
    qa_path: Path,
    index_dir: Path,
    backend: str = "visual_stub",
    top_k: int = 10,
) -> Dict[str, float]:
    examples = [item for item in _read_jsonl(Path(qa_path)) if item.get("answerable", True)]
    if not examples:
        return {
            "evaluated_count": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }

    recall_1 = 0
    recall_5 = 0
    mrr_10 = 0.0
    ndcg_10 = 0.0
    latencies = []

    for item in examples:
        start = time.perf_counter()
        ranked = visual_search(item["question"], index_dir=index_dir, backend=backend, top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        gold_sources = set(item.get("evidence_sources", []))
        gold_page_ids = set(item.get("evidence_page_ids", []))
        hit_positions = [
            idx + 1
            for idx, candidate in enumerate(ranked[:10])
            if _is_hit(candidate, gold_sources, gold_page_ids)
        ]
        if ranked[:1] and _is_hit(ranked[0], gold_sources, gold_page_ids):
            recall_1 += 1
        if any(_is_hit(candidate, gold_sources, gold_page_ids) for candidate in ranked[:5]):
            recall_5 += 1
        if hit_positions:
            first_hit = hit_positions[0]
            mrr_10 += 1.0 / first_hit
            ndcg_10 += 1.0 / math.log2(first_hit + 1)

    count = len(examples)
    latencies.sort()
    p50 = latencies[int(0.5 * (len(latencies) - 1))]
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    return {
        "evaluated_count": count,
        "recall_at_1": recall_1 / count,
        "recall_at_5": recall_5 / count,
        "mrr_at_10": mrr_10 / count,
        "ndcg_at_10": ndcg_10 / count,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }
