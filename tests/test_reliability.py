import unittest

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.qa import generate_unsupported_questions


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


if __name__ == "__main__":
    unittest.main()
