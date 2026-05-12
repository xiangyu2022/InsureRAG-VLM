import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_vlm_model() -> str:
    # Priority: Ollama (free local) > Anthropic > OpenAI > local-extractive
    # Ollama model is detected at VLMClient init; just pick a safe default name here.
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-haiku-4-5"
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return "local-extractive"


@dataclass
class ModelConfig:
    retrieval_model: str = field(default_factory=lambda: os.environ.get("OPENAI_EMBEDDING_MODEL", "local-hashing"))
    vlm_model: str = field(default_factory=_default_vlm_model)
    use_hf_api: bool = True
    hf_api_token: str | None = field(default_factory=lambda: os.environ.get("HF_API_TOKEN"))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY"))
    max_retrievals: int = 5
    max_answer_pages: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_MAX_ANSWER_PAGES", "3")))
    max_page_chars: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_MAX_PAGE_CHARS", "900")))
    max_context_chars: int = field(default_factory=lambda: int(os.environ.get("INSURERAG_MAX_CONTEXT_CHARS", "2400")))
    abstain_threshold: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_ABSTAIN_THRESHOLD", "0.20")))
    citation_min_overlap: float = field(default_factory=lambda: float(os.environ.get("INSURERAG_CITATION_MIN_OVERLAP", "0.20")))
    index_dir: Path = field(default_factory=lambda: Path("data"))
    index_path: Path | None = None
    metadata_path: Path | None = None
    render_pdf_pages: bool = False
    pdf_render_dir: Path = field(default_factory=lambda: Path("data/pdf_pages"))
    prompt_template: str = (
        "You are a citation-grounded insurance policy assistant. "
        "Use the retrieved page snippets to answer the user's question. "
        "Cite the source pages and abstain if the answer is not supported.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    def __post_init__(self) -> None:
        self.index_path = self.index_path or self.index_dir / "index.npy"
        self.metadata_path = self.metadata_path or self.index_dir / "index_meta.json"
        self.pdf_render_dir = self.pdf_render_dir or self.index_dir / "pdf_pages"
