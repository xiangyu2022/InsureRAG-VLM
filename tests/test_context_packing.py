import tempfile
import unittest
from pathlib import Path

import fitz

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.context_packing import pack_evidence
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


class ContextPackingTests(unittest.TestCase):
    def test_long_first_source_cannot_swallow_other_sources(self):
        pages = [{"source": f"p{i}.pdf#page=1", "score": 10-i,
                  "text_snippet": (f"Evidence for page {i}. " * 100),
                  "snippet_support": [(f"Evidence for page {i}. " * 200)]}
                 for i in range(3)]
        packed = pack_evidence(pages, 3, max_chars=1200)
        self.assertEqual(packed["sources"], [p["source"] for p in pages])
        self.assertLessEqual(len(packed["context"]), 1200)
        for segment in packed["segments"]:
            page = next(p for p in pages if p["source"] == segment["source"])
            self.assertIn(segment["text"], page["text_snippet"])
        self.assertEqual(len(packed["segments"]), 3)

    def test_complete_prompt_capacity_and_source_ids_are_atomic(self):
        # Actual tokenizer coverage is exercised by the offline fixture script;
        # this deterministic capacity oracle stresses header/body allocation.
        pages = [{"source": "nested/unchanged-policy.pdf#page=12", "text_snippet": "Long evidence. "*500},
                 {"source": "second.pdf#page=3", "text_snippet": "Other evidence. "*500}]
        counter = lambda context: 100 + len(context.encode("utf-8"))
        packed = pack_evidence(pages, 2, prompt_token_counter=counter, max_prompt_tokens=400)
        self.assertLessEqual(packed["prompt_tokens"], 400)
        self.assertEqual(packed["sources"], [p["source"] for p in pages])
        self.assertIn("SOURCE: nested/unchanged-policy.pdf#page=12\n", packed["context"])
        with self.assertRaises(ValueError):
            pack_evidence(pages, 2, prompt_token_counter=counter, max_prompt_tokens=50)

    def test_scoped_query_does_not_select_navigation_instead_of_insurance_clause(self):
        sentence = "The collision deductible is $500 only when a covered collision loss occurs."
        pages = [{"source": "https://example.org/contact/insurance/guide", "text_snippet": "Navigation.",
                  "snippet_support": ["Skip to main content. Privacy policy. Contact the department at 800-123-4567. " + sentence]}]
        packed = pack_evidence(pages, 1, max_page_chars=120,
            question="In the document https://example.org/contact/insurance/guide, What is the collision deductible?")
        self.assertIn(sentence, packed["context"])
        self.assertNotIn("800-123", packed["context"])

    def test_actual_pdf_retrieval_keeps_relevant_page_before_low_ranked_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            for name, text in (("travel", "Travel insurance. Trip cancellation reimbursement requires a covered reason."),
                               ("auto", "Auto policy coverage. Collision deductible is $500.")):
                with fitz.open() as pdf:
                    pdf.new_page().insert_text((40, 40), text)
                    pdf.save(corpus / f"{name}.pdf")
            pipeline = DocumentRetrievalPipeline(ModelConfig(index_dir=root / "index",
                corpus_source="documents", retrieval_model="local-hashing", vlm_model="local-extractive",
                use_hf_api=False, enable_image_signal=False, max_context_chars=700))
            pipeline.build_index(corpus)
            ranked = pipeline.rank_pages("What does travel trip cancellation insurance require?", corpus, top_k=2)
            self.assertTrue(ranked[0]["source"].startswith("travel.pdf"))
            packed = pipeline.pack_context_with_audit(ranked, 2)
            self.assertEqual(packed["sources"][0], ranked[0]["source"])
            self.assertIn("covered reason", packed["context"])
            # A irrelevant role is not permitted to overturn an established rank.
            ranked[0]["primary_clause_type"] = "definition"
            ranked[1]["primary_clause_type"] = "coverage"
            packed = pipeline.pack_context_with_audit(ranked, 1)
            self.assertEqual(packed["sources"], [ranked[0]["source"]])
            self.assertNotIn("$500", packed["context"])


if __name__ == "__main__":
    unittest.main()
