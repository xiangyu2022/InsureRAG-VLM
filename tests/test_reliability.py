import unittest

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import compute_retrieval_metrics, generate_unsupported_questions


class UnsupportedQuestionTests(unittest.TestCase):
    def test_default_unsupported_question_set_is_large_and_unique(self):
        questions = generate_unsupported_questions()

        self.assertGreaterEqual(len(questions), 50)
        self.assertEqual(len(questions), len(set(questions)))


class CitationValidationTests(unittest.TestCase):
    def test_rejects_answer_amount_not_present_in_citation(self):
        supported, reason = DocumentRetrievalPipeline._citation_support_details(
            "What deductible is stated?",
            "The deductible is $5,000.",
            [{"evidence_text": "The deductible is $1,000."}],
        )

        self.assertFalse(supported)
        self.assertEqual(reason, "answer_amount_not_in_citation")

    def test_accepts_answer_supported_by_citation(self):
        supported, reason = DocumentRetrievalPipeline._citation_support_details(
            "What deductible is stated?",
            "The deductible is $1,000.",
            [{"evidence_text": "The policy states that the deductible is $1,000."}],
        )

        self.assertTrue(supported)
        self.assertEqual(reason, "supported")


class ConfigTests(unittest.TestCase):
    def test_abstain_threshold_has_safe_default(self):
        self.assertEqual(ModelConfig().abstain_threshold, 0.20)

    def test_hybrid_multimodal_is_default_retrieval_mode(self):
        self.assertEqual(ModelConfig().retrieval_mode, "hybrid_multimodal")


class RetrievalMetricNormalizationTests(unittest.TestCase):
    def test_retrieval_metrics_match_page_key_when_url_page_fragment_differs(self):
        class StubPipeline:
            def rank_pages(self, question, data_folder, top_k=10):
                return [
                    {
                        "source": "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx",
                        "page_key": "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx::p0001",
                    }
                ]

        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            qa_path = Path(tmpdir) / "retrieval_eval.jsonl"
            qa_path.write_text(
                json.dumps(
                    {
                        "question": "What does replacement cost mean in this insurance context?",
                        "answerable": True,
                        "evidence_sources": [
                            "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx#page=1"
                        ],
                        "gold_page_keys": [
                            "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx::p0001"
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metrics = compute_retrieval_metrics(StubPipeline(), Path("."), qa_path, top_k=10)

        self.assertEqual(metrics.evaluated_count, 1)
        self.assertEqual(metrics.recall_at_1, 1.0)
        self.assertEqual(metrics.recall_at_5, 1.0)
        self.assertEqual(metrics.mrr_at_10, 1.0)


if __name__ == "__main__":
    unittest.main()
