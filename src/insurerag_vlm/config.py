from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    # Retrieval model for embeddings.
    retrieval_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # VLM model name for remote API calls.
    vlm_model: str = "Qwen2.5-VL-7B-Instruct"
    use_hf_api: bool = True
    hf_api_token: str | None = None
    openai_api_key: str | None = None
    max_retrievals: int = 5
    index_dir: Path = Path("data")
    index_path: Path | None = None
    metadata_path: Path | None = None
    render_pdf_pages: bool = False
    pdf_render_dir: Path = Path("data/pdf_pages")
    prompt_template: str = (
        "You are a citation-grounded insurance policy assistant. Use the retrieved page snippets and image OCR text to answer the user's question. "
        "Cite the source pages in the answer and abstain if the answer is unknown.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    def __post_init__(self) -> None:
        self.index_path = self.index_path or self.index_dir / "index.npy"
        self.metadata_path = self.metadata_path or self.index_dir / "index_meta.json"
        self.pdf_render_dir = self.pdf_render_dir or self.index_dir / "pdf_pages"
