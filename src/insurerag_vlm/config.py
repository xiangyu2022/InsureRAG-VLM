import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_vlm_model() -> str:
    # API credentials and installed models must never change the experimental arm.
    # Explicit local LLM example: INSURERAG_VLM_MODEL=ollama:qwen3.5:4b.
    return os.environ.get("INSURERAG_VLM_MODEL", "local-extractive")


def _optional_retrieval_length() -> int | None:
    value = os.environ.get("INSURERAG_RETRIEVAL_MAX_LENGTH")
    return int(value) if value else None


@dataclass
class ModelConfig:
    retrieval_model: str = field(default_factory=lambda: os.environ.get("INSURERAG_RETRIEVAL_MODEL", "local-hashing"))
    retrieval_pooling: str = field(default_factory=lambda: os.environ.get("INSURERAG_RETRIEVAL_POOLING", "auto"))
    retrieval_query_instruction: str = field(default_factory=lambda: os.environ.get("INSURERAG_RETRIEVAL_QUERY_INSTRUCTION", ""))
    retrieval_max_length: int | None = field(default_factory=_optional_retrieval_length)
    vlm_model: str = field(default_factory=_default_vlm_model)
    vlm_provider: str | None = None
    ollama_base_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_generation_options: dict = field(default_factory=dict)
    vlm_thinking: bool = False
    vlm_request_timeout: float = 300
    vlm_expected_digest: str | None = None
    use_hf_api: bool = True
    hf_api_token: str | None = field(default_factory=lambda: os.environ.get("HF_API_TOKEN"))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    retrieval_mode: str = field(default_factory=lambda: os.environ.get("INSURERAG_RETRIEVAL_MODE", "hybrid_multimodal"))
    corpus_source: str = field(default_factory=lambda: os.environ.get("INSURERAG_CORPUS_SOURCE", "auto"))
    enable_image_signal: bool = field(default_factory=lambda: os.environ.get("INSURERAG_ENABLE_IMAGE_SIGNAL", "1") != "0")
    image_signal_weight: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_IMAGE_SIGNAL_WEIGHT", "0.10")))
    dense_weight: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_DENSE_WEIGHT", "0.55")))
    sparse_weight: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_SPARSE_WEIGHT", "0.45")))
    merge_method: str = field(default_factory=lambda: os.environ.get("INSURERAG_MERGE_METHOD", "rrf"))
    rrf_k: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_RRF_K", "60")))
    snippet_top_k: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_SNIPPET_TOP_K", "24")))
    page_top_k: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_PAGE_TOP_K", "12")))
    max_retrievals: int = 5
    max_answer_pages: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_MAX_ANSWER_PAGES", "3")))
    max_page_chars: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_MAX_PAGE_CHARS", "2400")))
    max_context_chars: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_MAX_CONTEXT_CHARS", "8000")))
    candidate_pool_size: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_CANDIDATE_POOL_SIZE", "40")))
    graph_mode: str = field(default_factory=lambda: os.environ.get("INSURERAG_GRAPH_MODE", "explicit"))
    graph_seed_pages: int = 4
    graph_max_hops: int = 2
    graph_max_expansions: int = 8
    abstain_threshold: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_ABSTAIN_THRESHOLD", "0.20")))
    citation_min_overlap: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_CITATION_MIN_OVERLAP", "0.20")))
    index_dir: Path = field(default_factory=lambda: Path("data"))
    index_path: Path | None = None
    metadata_path: Path | None = None
    curated_dataset_dir: Path = field(default_factory=lambda: Path(os.environ.get("INSURERAG_CURATED_DATASET_DIR", "data/04_curated")))
    render_pdf_pages: bool = False
    pdf_render_dir: Path = field(default_factory=lambda: Path("data/pdf_pages"))
    prompt_template: str = (
        "You are a citation-grounded insurance policy assistant. "
        "Use the retrieved page snippets to answer the user's question. "
        "Cite the source pages and abstain if the answer is not supported. "
        "A person's own policy terms require evidence from the relevant policy, declarations, or endorsement; "
        "a public guide or illustrative example cannot establish their own coverage, even when uploaded.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    def __post_init__(self) -> None:
        if self.graph_mode not in {"off", "explicit", "all"}:
            raise ValueError("graph_mode must be off, explicit, or all")
        if self.graph_seed_pages < 1 or self.graph_max_hops < 0 or self.graph_max_expansions < 0:
            raise ValueError("Invalid graph traversal budgets")
        self.index_path = self.index_path or self.index_dir / "index.npy"
        self.metadata_path = self.metadata_path or self.index_dir / "index_meta.json"
        self.pdf_render_dir = self.pdf_render_dir or self.index_dir / "pdf_pages"
