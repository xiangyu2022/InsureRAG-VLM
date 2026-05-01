import json
import re
from pathlib import Path
from typing import Dict, List, Optional

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
            anthropic_api_key=getattr(config, "anthropic_api_key", None),
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

    def query_with_ranking(
        self,
        question: str,
        data_folder: Path,
        top_k: Optional[int] = None,
        force_extractive: bool = False,
    ) -> Dict[str, object]:
        top_k = top_k or self.config.max_retrievals
        answer_top_k = min(top_k, getattr(self.config, "max_answer_pages", top_k))
        index = load_index(self.config.index_path)

        candidates = self.retriever.search(question, index, top_k=top_k, return_scores=True)
        documents = self._load_documents(data_folder)

        context_parts = []
        ranked_pages = []
        for rank, (idx, score) in enumerate(candidates):
            if idx >= len(documents):
                continue
            doc = documents[idx]
            source = doc.metadata.get("source", doc.doc_id)
            evidence_snippet = self._select_evidence_snippet(
                question,
                doc.text,
                max_chars=getattr(self.config, "max_page_chars", 900),
            )
            ranked_pages.append({
                "source": source,
                "score": score,
                "page_number": doc.page_number,
                "text_snippet": evidence_snippet,
            })
            if rank < answer_top_k:
                context_parts.append(f"SOURCE: {source}\n{evidence_snippet}\n")

        combined_context = self._trim_context(
            "\n---\n".join(context_parts),
            max_chars=getattr(self.config, "max_context_chars", 2400),
        )
        prompt = format_prompt(combined_context, question, self.config.prompt_template)
        answer = self.vlm_client.generate_extractive(prompt) if force_extractive else self.vlm_client.generate(prompt)
        return {"answer": answer, "source_ranking": ranked_pages}

    def query_structured(
        self,
        question: str,
        data_folder: Path,
        top_k: Optional[int] = None,
        force_extractive: bool = False,
    ) -> Dict[str, object]:
        result = self.query_with_ranking(question, data_folder, top_k=top_k, force_extractive=force_extractive)
        answer = str(result["answer"]).strip()
        ranked_pages = result["source_ranking"]
        citations = []
        cited_source = self._extract_answer_source(answer, ranked_pages)
        clean_answer = re.sub(r"\n+\s*SOURCE:\s*.*$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
        cited_page = None
        if cited_source:
            cited_page = next((page for page in ranked_pages if page["source"] == cited_source), None)
        if cited_page is None:
            cited_page = self._choose_cited_page(question, clean_answer, ranked_pages)
        if cited_page:
            citations.append(
                {
                    "source": cited_page["source"],
                    "page_id": str(cited_page["source"]).replace("/", "_").replace("#page=", "_p"),
                    "evidence_text": cited_page.get("text_snippet", ""),
                }
            )

        confidence = self._estimate_confidence(question, clean_answer, ranked_pages)
        abstain = confidence < 0.20 or "insufficient_evidence" in answer.lower()
        return {
            "answer": "" if abstain else clean_answer,
            "citations": [] if abstain else citations,
            "confidence": confidence,
            "abstain": abstain,
            "abstain_reason": "insufficient_retrieved_evidence" if abstain else None,
            "source_ranking": ranked_pages,
        }

    @staticmethod
    def _extract_answer_source(answer: str, ranked_pages: Optional[List[Dict[str, object]]] = None) -> Optional[str]:
        match = re.search(r"SOURCE:\s*(.+)", answer, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        for page in ranked_pages or []:
            source = str(page.get("source", ""))
            if source and source in answer:
                return source
        return None

    @staticmethod
    def _choose_cited_page(
        question: str,
        answer: str,
        ranked_pages: List[Dict[str, object]],
    ) -> Optional[Dict[str, object]]:
        if not ranked_pages:
            return None
        answer_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9$%]+", answer.lower())
            if len(term) > 2
        }
        question_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9$%]+", question.lower())
            if len(term) > 2
        }
        answer_amounts = set(re.findall(r"\$[\d,]+(?:\.\d+)?", answer))
        best_page = ranked_pages[0]
        best_score = -1.0
        for page in ranked_pages:
            text = str(page.get("text_snippet", ""))
            text_terms = set(re.findall(r"[a-zA-Z0-9$%]+", text.lower()))
            text_amounts = set(re.findall(r"\$[\d,]+(?:\.\d+)?", text))
            score = float(len(answer_terms & text_terms))
            score += 0.5 * len(question_terms & text_terms)
            score += 4.0 * len(answer_amounts & text_amounts)
            score += 0.1 * float(page.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_page = page
        return best_page

    @staticmethod
    def _estimate_confidence(question: str, answer: str, ranked_pages: List[Dict[str, object]]) -> float:
        if not ranked_pages:
            return 0.0
        top_score = max(0.0, min(1.0, float(ranked_pages[0].get("score", 0.0))))
        question_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", question.lower()) if len(term) > 2}
        answer_terms = {term for term in re.findall(r"[a-zA-Z0-9$%]+", answer.lower()) if len(term) > 2}
        overlap = len(question_terms & answer_terms) / max(1, len(question_terms))
        return round(0.7 * top_score + 0.3 * overlap, 4)

    @staticmethod
    def _select_evidence_snippet(question: str, text: str, max_chars: int = 900) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= max_chars:
            return text

        question_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9$%]+", question.lower())
            if len(term) > 2
        }
        generic_terms = {
            "what", "which", "does", "the", "this", "that", "policy", "coverage",
            "insured", "provide", "provides", "apply", "applies", "listed",
        }
        key_terms = question_terms - generic_terms
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        scored = []
        for position, sentence in enumerate(sentences):
            sentence_terms = set(re.findall(r"[a-zA-Z0-9$%]+", sentence.lower()))
            score = len(question_terms & sentence_terms)
            if "$" in sentence:
                score += 1
            if "deductible" in question_terms and "deductible" in sentence_terms:
                score += 2
            if {"limit", "limits"} & question_terms and {"limit", "limits"} & sentence_terms:
                score += 2
            if key_terms and not (key_terms & sentence_terms):
                score -= 1
            scored.append((score, position, sentence))

        selected = []
        total_chars = 0
        for score, position, sentence in sorted(scored, key=lambda item: (-item[0], item[1])):
            if score <= 0 and selected:
                break
            if total_chars + len(sentence) + 1 > max_chars:
                continue
            selected.append((position, sentence))
            total_chars += len(sentence) + 1
            if total_chars >= max_chars * 0.75:
                break

        if not selected:
            return text[:max_chars].rstrip()
        selected_text = " ".join(sentence for _, sentence in sorted(selected))
        return selected_text[:max_chars].rstrip()

    @staticmethod
    def _trim_context(context: str, max_chars: int = 2400) -> str:
        if len(context) <= max_chars:
            return context
        return context[:max_chars].rsplit("\n---\n", 1)[0].strip() or context[:max_chars].rstrip()

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
