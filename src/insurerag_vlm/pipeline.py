import json
from pathlib import Path
from typing import List, Optional

from .config import ModelConfig
from .data import PageDocument, load_documents
from .evaluation import evaluate_predictions, load_evaluation_examples
from .ocr import extract_text_from_image
from .retriever import EmbeddingRetriever, load_index
from .vlm import VLMClient, format_prompt


class DocumentRetrievalPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.retriever = EmbeddingRetriever(
            config.retrieval_model,
            use_hf_api=config.use_hf_api,
            hf_api_token=config.hf_api_token,
            openai_api_key=config.openai_api_key,
        )
        self.vlm_client = VLMClient(
            model_name=config.vlm_model,
            hf_api_token=config.hf_api_token,
            openai_api_key=config.openai_api_key,
            use_hf_api=config.use_hf_api,
        )

    def build_index(self, data_folder: Path) -> None:
        documents = load_documents(
            data_folder,
            render_pdf_pages=self.config.render_pdf_pages,
            pdf_render_dir=self.config.pdf_render_dir,
        )
        enriched_documents = []
        for doc in documents:
            if doc.image_path and not doc.text:
                doc.text = extract_text_from_image(doc.image_path)
            enriched_documents.append(doc)

        serialized_documents = [
            {"text": doc.text, "metadata": doc.metadata}
            for doc in enriched_documents
        ]
        self.retriever.build_index(serialized_documents, self.config.index_path, self.config.metadata_path)

    def _load_documents(self, data_folder: Path) -> List[PageDocument]:
        documents = load_documents(
            data_folder,
            render_pdf_pages=self.config.render_pdf_pages,
            pdf_render_dir=self.config.pdf_render_dir,
        )
        for doc in documents:
            if doc.image_path and not doc.text:
                doc.text = extract_text_from_image(doc.image_path)
        return documents

    def query(self, question: str, data_folder: Path, top_k: Optional[int] = None) -> str:
        result = self.query_with_ranking(question, data_folder, top_k=top_k)
        return result["answer"]

    def query_with_ranking(self, question: str, data_folder: Path, top_k: Optional[int] = None) -> Dict[str, object]:
        top_k = top_k or self.config.max_retrievals
        index = load_index(self.config.index_path)

        candidates = self.retriever.search(question, index, top_k=top_k, return_scores=True)
        documents = self._load_documents(data_folder)

        context_parts = []
        ranked_pages = []
        for idx, score in candidates:
            if idx >= len(documents):
                continue
            doc = documents[idx]
            source = doc.metadata.get("source", doc.doc_id)
            ranked_pages.append({
                "source": source,
                "score": score,
                "page_number": doc.page_number,
                "text_snippet": doc.text[:300],
            })
            context_parts.append(f"SOURCE: {source}\n{doc.text}\n")

        combined_context = "\n---\n".join(context_parts)
        prompt = format_prompt(combined_context, question, self.config.prompt_template)
        answer = self.vlm_client.generate(prompt)
        return {"answer": answer, "source_ranking": ranked_pages}

    def rank_pages(self, question: str, data_folder: Path, top_k: Optional[int] = None) -> List[Dict[str, object]]:
        top_k = top_k or self.config.max_retrievals
        index = load_index(self.config.index_path)
        candidates = self.retriever.search(question, index, top_k=top_k, return_scores=True)
        documents = self._load_documents(data_folder)

        ranked_pages = []
        for idx, score in candidates:
            if idx >= len(documents):
                continue
            doc = documents[idx]
            ranked_pages.append({
                "source": doc.metadata.get("source", doc.doc_id),
                "score": score,
                "page_number": doc.page_number,
                "text_snippet": doc.text[:300],
                "image_path": str(doc.image_path) if doc.image_path else None,
            })
        return ranked_pages

    def evaluate(self, data_folder: Path, examples_path: Path, top_k: Optional[int] = None):
        if not self.config.index_path.exists() or not self.config.metadata_path.exists():
            self.build_index(data_folder)

        from .evaluation import evaluate_predictions, load_evaluation_examples

        examples = load_evaluation_examples(examples_path)
        predictions = []
        for example in examples:
            prediction = self.query(example.question, data_folder, top_k=top_k)
            predictions.append((example.question, prediction))

        return evaluate_predictions(predictions, examples)

    def quick_demo(self, data_folder: Path) -> None:
        print("Building or reusing the index from:", self.config.index_path)
        if not self.config.index_path.exists() or not self.config.metadata_path.exists():
            print("Index files not found. Building new index...")
            self.build_index(data_folder)
        else:
            print("Index already exists. Skipping build.")

        while True:
            question = input("Enter a question about the insurance documents (or 'quit'): ").strip()
            if not question or question.lower() in {"quit", "exit"}:
                break
            answer = self.query(question, data_folder)
            print("\n=== ANSWER ===\n", answer)

