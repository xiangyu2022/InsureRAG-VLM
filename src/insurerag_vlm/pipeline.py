import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from .config import ModelConfig
from .data import PageDocument, load_documents
from .evaluation import evaluate_predictions, load_evaluation_examples
from .ocr import extract_text_from_image
from .retriever import EmbeddingRetriever, load_index
from .vlm import VLMClient, format_prompt


_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d+/\d+/\d+\b")


class DocumentRetrievalPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._documents_cache: Dict[tuple[str, bool, str], List[PageDocument]] = {}
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
        cache_key = (
            str(Path(data_folder).resolve()),
            bool(self.config.render_pdf_pages),
            str(Path(self.config.pdf_render_dir).resolve()) if self.config.pdf_render_dir else "",
        )
        if cache_key in self._documents_cache:
            return self._documents_cache[cache_key]
        documents = load_documents(
            data_folder,
            render_pdf_pages=self.config.render_pdf_pages,
            pdf_render_dir=self.config.pdf_render_dir,
        )
        for doc in documents:
            if doc.image_path and not doc.text:
                doc.text = extract_text_from_image(doc.image_path)
        self._documents_cache[cache_key] = documents
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
        if index.size == 0:
            return {
                "answer": "I cannot support an answer from the retrieved evidence. SOURCE: insufficient_evidence",
                "source_ranking": [],
            }

        candidate_pool = min(len(index), max(top_k, top_k * 4, getattr(self.config, "candidate_pool_size", 20)))
        candidates = self.retriever.search(question, index, top_k=candidate_pool, return_scores=True)
        documents = self._load_documents(data_folder)

        ranked_pages = []
        for idx, score in candidates:
            if idx >= len(documents):
                continue
            doc = documents[idx]
            source = doc.metadata.get("source", doc.doc_id)
            evidence_snippet = self._select_evidence_snippet(
                question,
                doc.text,
                max_chars=getattr(self.config, "max_page_chars", 900),
            )
            rerank_score = self._score_insurance_evidence(question, doc.text, source, doc.page_number)
            ranked_pages.append({
                "source": source,
                "score": round(float(score) + rerank_score, 6),
                "retrieval_score": score,
                "rerank_score": round(rerank_score, 6),
                "page_number": doc.page_number,
                "text_snippet": evidence_snippet,
            })

        ranked_pages.sort(key=lambda page: float(page.get("score", 0.0)), reverse=True)
        ranked_pages = ranked_pages[:top_k]

        context_parts = []
        for rank, page in enumerate(ranked_pages):
            if rank < answer_top_k:
                context_parts.append(f"SOURCE: {page['source']}\n{page['text_snippet']}\n")

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
            repaired_answer = self._repair_answer_from_evidence(question, clean_answer, cited_page)
            if not repaired_answer and (
                "insufficient_evidence" in answer.lower() or not clean_answer
            ):
                repaired_answer = self._best_sentence_from_evidence(
                    question,
                    str(cited_page.get("text_snippet", "")),
                )
            if repaired_answer:
                clean_answer = repaired_answer
            citations.append(
                {
                    "source": cited_page["source"],
                    "page_id": str(cited_page["source"]).replace("/", "_").replace("#page=", "_p"),
                    "evidence_text": cited_page.get("text_snippet", ""),
                }
            )

        confidence = self._estimate_confidence(question, clean_answer, ranked_pages)
        supported, support_reason = self._citation_support_details(
            question,
            clean_answer,
            citations,
            min_overlap=getattr(self.config, "citation_min_overlap", 0.20),
        )
        if not supported:
            confidence = min(confidence, 0.19)
        threshold = getattr(self.config, "abstain_threshold", 0.20)
        abstain = confidence < threshold or "insufficient_evidence" in answer.lower() or not supported
        return {
            "answer": "" if abstain else clean_answer,
            "citations": [] if abstain else citations,
            "confidence": confidence,
            "abstain": abstain,
            "abstain_reason": "insufficient_retrieved_evidence" if abstain else None,
            "citation_support": supported,
            "citation_support_reason": support_reason,
            "source_ranking": ranked_pages,
        }

    @staticmethod
    def _terms(text: str, min_len: int = 2) -> Set[str]:
        return {term for term in re.findall(r"[a-zA-Z0-9$%]+", text.lower()) if len(term) >= min_len}

    @staticmethod
    def _generic_terms() -> Set[str]:
        return {
            "what", "which", "does", "this", "that", "the", "policy", "coverage",
            "include", "includes", "provide", "provides", "listed", "insurance",
            "from", "your", "about", "page", "evidence", "mentioning", "explains",
            "summarize", "guidance", "consumer", "know", "passage", "related",
            "described", "find", "amount", "numeric", "detail", "stated", "for",
            "and", "after", "before", "with", "into", "under", "document", "guide",
            "pdf",
        }

    @classmethod
    def _key_terms(cls, text: str) -> Set[str]:
        return cls._terms(text, min_len=3) - cls._generic_terms()

    @staticmethod
    def _question_requires_numeric_evidence(question: str) -> bool:
        lowered = question.lower()
        numeric_intents = {
            "amount", "limit", "limits", "deductible", "premium", "sublimit",
            "coinsurance", "retention", "per person", "per accident", "per day",
            "maximum", "minimum", "how much", "dollar", "percent", "percentage",
        }
        return any(intent in lowered for intent in numeric_intents)

    @staticmethod
    def _insurance_field_groups(question: str) -> List[Set[str]]:
        lowered = question.lower()
        groups: List[Set[str]] = []
        if "liability" in lowered:
            groups.append({"liability"})
        if "comprehensive" in lowered:
            groups.append({"comprehensive"})
        if "collision" in lowered:
            groups.append({"collision"})
        if "deductible" in lowered:
            groups.append({"deductible"})
        if "limit" in lowered or "limits" in lowered or "sublimit" in lowered:
            groups.append({"limit", "limits", "sublimit"})
        if "premium" in lowered:
            groups.append({"premium"})
        if "medical payments" in lowered:
            groups.append({"medical", "payments"})
        if "endorsement" in lowered or "rider" in lowered:
            groups.append({"endorsement", "rider"})
        if "exclusion" in lowered or "excludes" in lowered:
            groups.append({"exclusion", "exclusions", "excludes"})
        if "duties" in lowered or "after a loss" in lowered:
            groups.append({"duties", "loss", "notify", "cooperate"})
        return groups

    @staticmethod
    def _candidate_sentences(text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return []
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=:)\s+|\n+", normalized) if s.strip()]

    @classmethod
    def _score_insurance_evidence(
        cls,
        question: str,
        text: str,
        source: object = "",
        page_number: object = None,
    ) -> float:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return 0.0
        lowered = cleaned.lower()
        text_terms = cls._terms(cleaned, min_len=2)
        question_terms = cls._key_terms(question)
        field_groups = cls._insurance_field_groups(question)
        requires_numeric = cls._question_requires_numeric_evidence(question)

        score = 0.025 * len(question_terms & text_terms)
        for group in field_groups:
            score += 0.12 if group & text_terms else -0.08

        if requires_numeric:
            has_amount = bool(_AMOUNT_RE.search(cleaned))
            score += 0.25 if has_amount else -0.25
            if ("declaration" in lowered or "declarations" in lowered) and has_amount:
                score += 0.18
            if re.search(r"(limit|deductible|premium|sublimit)\s*:", lowered):
                score += 0.20
            if page_number in {1, "1"} and has_amount:
                score += 0.05

        for sentence in cls._candidate_sentences(cleaned):
            sentence_terms = cls._terms(sentence, min_len=2)
            if requires_numeric and _AMOUNT_RE.search(sentence):
                covered_groups = sum(1 for group in field_groups if group & sentence_terms)
                if covered_groups:
                    score += 0.18 * covered_groups
                if question_terms and len(question_terms & sentence_terms) >= min(2, len(question_terms)):
                    score += 0.10

        return round(score, 6)

    @classmethod
    def _repair_answer_from_evidence(
        cls,
        question: str,
        answer: str,
        cited_page: Dict[str, object],
    ) -> Optional[str]:
        if not cls._question_requires_numeric_evidence(question):
            return None
        if set(_AMOUNT_RE.findall(answer or "")):
            return None
        evidence = str(cited_page.get("text_snippet", ""))
        field_groups = cls._insurance_field_groups(question)
        best_sentence = ""
        best_score = 0
        for sentence in cls._candidate_sentences(evidence):
            if not _AMOUNT_RE.search(sentence):
                continue
            sentence_terms = cls._terms(sentence, min_len=2)
            score = sum(1 for group in field_groups if group & sentence_terms)
            if "limit" in sentence_terms or "deductible" in sentence_terms:
                score += 1
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence if best_score > 0 else None

    @classmethod
    def _best_sentence_from_evidence(
        cls,
        question: str,
        evidence: str,
    ) -> Optional[str]:
        question_terms = cls._key_terms(question)
        if not question_terms:
            return None
        best_sentence = ""
        best_score = 0.0
        for sentence in cls._candidate_sentences(evidence):
            sentence_terms = cls._terms(sentence, min_len=3)
            overlap = question_terms & sentence_terms
            if not overlap:
                continue
            score = float(len(overlap))
            if _AMOUNT_RE.search(sentence):
                score += 1.0
            if len(sentence) < 40:
                score -= 0.5
            if score > best_score:
                best_score = score
                best_sentence = sentence
        return best_sentence if best_score >= 2.0 else None

    @classmethod
    def _answer_supported_by_citations(
        cls,
        question: str,
        answer: str,
        citations: List[Dict[str, object]],
    ) -> bool:
        supported, _ = cls._citation_support_details(question, answer, citations)
        return supported

    @classmethod
    def _citation_support_details(
        cls,
        question: str,
        answer: str,
        citations: List[Dict[str, object]],
        min_overlap: float = 0.20,
    ) -> tuple[bool, str]:
        if "insufficient_evidence" in (answer or "").lower():
            return False, "model_reported_insufficient_evidence"
        if not citations:
            return False, "missing_citation"
        evidence = " ".join(str(citation.get("evidence_text", "")) for citation in citations)
        if not evidence.strip():
            return False, "missing_citation_evidence"

        answer_amounts = set(_AMOUNT_RE.findall(answer or ""))
        evidence_amounts = set(_AMOUNT_RE.findall(evidence))
        if answer_amounts and not answer_amounts <= evidence_amounts:
            return False, "answer_amount_not_in_citation"
        if cls._question_requires_numeric_evidence(question) and not answer_amounts:
            return False, "numeric_question_without_answer_amount"

        key_terms = cls._key_terms(question)
        evidence_terms = cls._terms(evidence, min_len=3)
        if key_terms and not (key_terms & evidence_terms):
            return False, "question_terms_not_in_citation"
        broad_value_terms = {
            "coverage", "cover", "include", "policy", "insurance", "limit", "limits",
            "sublimit", "deductible", "endorsement", "reimbursement", "provision",
            "amount", "liability", "property", "loss", "use", "auto", "automobile",
            "guide", "document", "pdf",
        }
        specific_terms = {term for term in key_terms if len(term) >= 4 and term not in broad_value_terms}
        if specific_terms and not specific_terms <= evidence_terms:
            return False, "specific_question_terms_not_in_citation"

        answer_terms = cls._key_terms(answer)
        answer_specific_terms = {
            term for term in answer_terms
            if len(term) >= 4 and term not in broad_value_terms
        }
        if answer_specific_terms:
            overlap_ratio = len(answer_specific_terms & evidence_terms) / max(1, len(answer_specific_terms))
            if overlap_ratio < min_overlap:
                return False, "answer_terms_not_supported_by_citation"
        return True, "supported"

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
        answer_amounts = set(_AMOUNT_RE.findall(answer))
        best_page = ranked_pages[0]
        best_score = -1.0
        for page in ranked_pages:
            text = str(page.get("text_snippet", ""))
            text_terms = set(re.findall(r"[a-zA-Z0-9$%]+", text.lower()))
            text_amounts = set(_AMOUNT_RE.findall(text))
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
        generic_terms = DocumentRetrievalPipeline._generic_terms()
        key_terms = question_terms - generic_terms
        evidence_terms = {
            term
            for page in ranked_pages[:3]
            for term in re.findall(r"[a-zA-Z0-9$%]+", str(page.get("text_snippet", "")).lower())
            if len(term) > 2
        }
        overlap = len(question_terms & answer_terms) / max(1, len(question_terms))
        evidence_overlap = len(key_terms & evidence_terms) / max(1, len(key_terms))
        confidence = 0.55 * top_score + 0.20 * overlap + 0.25 * evidence_overlap
        missing_specific_terms = [
            term for term in key_terms if len(term) >= 7 and term not in evidence_terms
        ]
        if DocumentRetrievalPipeline._question_requires_numeric_evidence(question):
            top_evidence = " ".join(str(page.get("text_snippet", "")) for page in ranked_pages[:3])
            answer_amounts = set(_AMOUNT_RE.findall(answer))
            evidence_amounts = set(_AMOUNT_RE.findall(top_evidence))
            if not answer_amounts or not answer_amounts <= evidence_amounts:
                confidence = min(confidence, 0.19)
            else:
                confidence = max(confidence, 0.42)
        if missing_specific_terms and evidence_overlap <= 0.50:
            confidence = min(confidence, 0.19)
        elif key_terms and evidence_overlap < 0.25:
            confidence = min(confidence, 0.19)
        return round(confidence, 4)

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
        sentences = DocumentRetrievalPipeline._candidate_sentences(text)
        scored = []
        for position, sentence in enumerate(sentences):
            sentence_terms = set(re.findall(r"[a-zA-Z0-9$%]+", sentence.lower()))
            score = len(question_terms & sentence_terms)
            if _AMOUNT_RE.search(sentence):
                score += 1
            if DocumentRetrievalPipeline._question_requires_numeric_evidence(question) and _AMOUNT_RE.search(sentence):
                score += 2
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
        if index.size == 0:
            return []
        candidate_pool = min(len(index), max(top_k, top_k * 4, getattr(self.config, "candidate_pool_size", 20)))
        candidates = self.retriever.search(question, index, top_k=candidate_pool, return_scores=True)
        documents = self._load_documents(data_folder)

        ranked_pages = []
        for idx, score in candidates:
            if idx >= len(documents):
                continue
            doc = documents[idx]
            source = doc.metadata.get("source", doc.doc_id)
            rerank_score = self._score_insurance_evidence(question, doc.text, source, doc.page_number)
            ranked_pages.append({
                "source": source,
                "score": round(float(score) + rerank_score, 6),
                "retrieval_score": score,
                "rerank_score": round(rerank_score, 6),
                "page_number": doc.page_number,
                "text_snippet": self._select_evidence_snippet(question, doc.text, max_chars=300),
                "image_path": str(doc.image_path) if doc.image_path else None,
            })
        ranked_pages.sort(key=lambda page: float(page.get("score", 0.0)), reverse=True)
        return ranked_pages[:top_k]

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


from .hybrid_pipeline import DocumentRetrievalPipeline as _HybridDocumentRetrievalPipeline

LegacyDocumentRetrievalPipeline = DocumentRetrievalPipeline
DocumentRetrievalPipeline = _HybridDocumentRetrievalPipeline
