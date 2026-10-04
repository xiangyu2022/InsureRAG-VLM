import unittest
from pathlib import Path
from unittest.mock import patch

from src.insurerag_vlm.answer_safety import is_explicit_abstention
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.pipeline import LegacyDocumentRetrievalPipeline


def example_guide_page():
    return {
        "source": "consumer-guide.pdf#page=2", "score": 0.95,
        "page_number": 2, "document_type": "consumer_guide", "primary_clause_type": "general",
        "text_snippet": "For example, a collision deductible of $500 means you pay the first $500 of a covered claim. Your own deductible is listed in your declarations.",
        "table_fields": [{"field_type": "deductible", "field_name": "Collision Deductible", "field_value": "$500", "coverage_tags": ["collision"]}],
    }


class SemanticAbstentionTests(unittest.TestCase):
    def test_common_explicit_declines_are_detected_without_special_token(self):
        for text in (
            "The supplied guide does not state your own deductible.",
            "Your deductible cannot be determined from the supplied example.",
            "Your limit is not specified in this document.",
            "I cannot confirm your personal premium from the retrieved text.",
            "There is not enough information to answer this question.",
            "SOURCE: insufficient_evidence",
            "I can not determine the approved payment from this packet.",
            "It is impossible to determine which version was issued later.",
            "Both versions lack dates. Therefore, it is impossible to determine which one controls.",
            "The provided evidence does not support a definitive answer regarding the controlling equipment limit.",
            "The supplied context does not support an answer to this question.",
            "No blanket limit for all property is stated for policy SYN-F-001.",
            "**No blanket coverage limit is specified in this packet.**",
            "Based on the evidence provided, there is no information about earthquake coverage limits established for this policy.",
            "There is no information regarding the approved payment.",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_explicit_abstention(text))

    def test_ordinary_exclusions_and_known_zero_deductibles_are_not_declines(self):
        for text in (
            "The policy does not cover electronic data loss.",
            "This coverage has no deductible.",
            "The collision deductible is $500.",
            "The collision deductible is $500; the policy does not cover electronic data loss.",
            "No exclusions are stated for glass coverage, and the stated glass limit is $1,000.",
            "The provided evidence does not support a flood exclusion; it expressly lists flood coverage up to $10,000.",
            "It is not impossible to determine the deductible: the stated amount is $500.",
            "Do not assume it is impossible to determine the limit; the limit is $5,000.",
            "The stated dwelling limit is $535,000, although actual payment depends on covered causes of loss.",
            "The collision deductible is $500. There is no information about the annual premium.",
            "There is not no information about the deductible: it is $500.",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_explicit_abstention(text))

    def test_semantic_decline_cannot_be_repaired_into_general_guide_example(self):
        raw = "The supplied guide does not state your own deductible."
        for pipeline_type in (DocumentRetrievalPipeline, LegacyDocumentRetrievalPipeline):
            with self.subTest(pipeline=pipeline_type.__module__):
                pipeline = pipeline_type(ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing"))
                result = {"answer": raw, "source_ranking": [example_guide_page()],
                          "generation_used": True, "answer_backend": "Ollama · qwen3.5:4b", "backend_metadata": {}}
                with patch.object(pipeline, "query_with_ranking", return_value=result), \
                        patch.object(pipeline, "_repair_answer_from_evidence") as repair:
                    final = pipeline.query_structured("What is my own collision deductible?", Path("unused"))
                self.assertTrue(final["abstain"])
                self.assertTrue(final["explicit_abstention"])
                self.assertFalse(final["answer_repaired"])
                self.assertEqual(final["raw_answer"], raw)
                self.assertEqual(final["answer"], "")
                self.assertEqual(final["citations"], [])
                self.assertIsNone(final.get("limit"))
                repair.assert_not_called()

    def test_direct_numeric_repair_also_respects_explicit_decline(self):
        for pipeline_type in (DocumentRetrievalPipeline, LegacyDocumentRetrievalPipeline):
            self.assertIsNone(pipeline_type._repair_answer_from_evidence(
                "What is my collision deductible?",
                "The supplied guide does not state your own deductible.", example_guide_page(),
            ))

    def test_missing_blanket_limit_is_not_repaired_into_individual_item_limit(self):
        raw = "No blanket limit for all property is stated for this policy."
        page = example_guide_page()
        page["text_snippet"] = "The scheduled camera coverage limit is $8,000. No blanket limit for all property is stated."
        for pipeline_type in (DocumentRetrievalPipeline, LegacyDocumentRetrievalPipeline):
            with self.subTest(pipeline=pipeline_type.__module__):
                pipeline = pipeline_type(ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing"))
                result = {"answer": raw, "source_ranking": [page], "generation_used": True,
                          "answer_backend": "Ollama · qwen3.5:4b", "backend_metadata": {}}
                with patch.object(pipeline, "query_with_ranking", return_value=result), \
                        patch.object(pipeline, "_repair_answer_from_evidence") as repair:
                    final = pipeline.query_structured("What is the blanket coverage limit for all property?", Path("unused"))
                self.assertTrue(final["explicit_abstention"])
                self.assertTrue(final["abstain"])
                self.assertFalse(final["answer_repaired"])
                self.assertEqual(final["raw_answer"], raw)
                self.assertEqual(final["answer"], "")
                self.assertEqual(final["citations"], [])
                repair.assert_not_called()

    def test_truncated_generation_is_not_accepted_or_repaired_even_with_matching_number(self):
        raw = "The collision deductible is $500.\n\nSOURCE: consumer-guide.pdf#page=2"
        for pipeline_type in (DocumentRetrievalPipeline, LegacyDocumentRetrievalPipeline):
            with self.subTest(pipeline=pipeline_type.__module__):
                pipeline = pipeline_type(ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing"))
                result = {"answer": raw, "source_ranking": [example_guide_page()],
                          "generation_used": True, "answer_backend": "Ollama · qwen3.5:4b",
                          "backend_metadata": {"last_generation": {"truncated": True, "done_reason": "length"}}}
                with patch.object(pipeline, "query_with_ranking", return_value=result), \
                        patch.object(pipeline, "_repair_answer_from_evidence") as repair:
                    final = pipeline.query_structured("What is the collision deductible?", Path("unused"))
                self.assertTrue(final["abstain"])
                self.assertTrue(final["generation_truncated"])
                self.assertEqual(final["abstain_reason"], "generation_truncated")
                self.assertEqual(final["raw_answer"], raw)
                self.assertEqual(final["answer"], "")
                self.assertFalse(final["answer_repaired"])
                repair.assert_not_called()

    def test_leading_no_information_response_does_not_trigger_numeric_repair(self):
        raw = "Based on the evidence provided, there is no information about earthquake coverage limits established for this policy."
        page = example_guide_page()
        page["text_snippet"] = "Water backup coverage limit is $4,600."
        pipeline = DocumentRetrievalPipeline(ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing"))
        result = {"answer": raw, "source_ranking": [page], "generation_used": True, "backend_metadata": {}}
        with patch.object(pipeline, "query_with_ranking", return_value=result), patch.object(pipeline, "_repair_answer_from_evidence") as repair:
            final = pipeline.query_structured("What earthquake coverage limit is established?", Path("unused"))
        assert final["explicit_abstention"] and final["abstain"]
        assert not final["answer_repaired"] and final["citation_origin"] is None
        repair.assert_not_called()


class StructuredNumericFieldTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = DocumentRetrievalPipeline(ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing"))

    def test_unrelated_first_table_value_is_not_reported_as_deductible(self):
        page = example_guide_page()
        page["table_fields"] = [{"field_type": "premium", "field_value": "$1,200"}]
        self.assertIsNone(self.pipeline._extract_structured_limit("What is the collision deductible?", page))

    def test_unstructured_first_amount_is_not_assumed_to_be_policy_field(self):
        page = example_guide_page()
        page["table_fields"] = []
        self.assertIsNone(self.pipeline._extract_structured_limit("What is my collision deductible?", page))

    def test_other_coverage_deductible_is_not_substituted_when_target_absent(self):
        page = example_guide_page()
        page["table_fields"] = [{"field_type": "deductible", "field_value": "$750", "coverage_tags": ["comprehensive"]}]
        self.assertIsNone(self.pipeline._extract_structured_limit("What is the collision deductible?", page))

    def test_matching_field_and_coverage_are_preserved(self):
        self.assertEqual(self.pipeline._extract_structured_limit("What is the collision deductible?", example_guide_page()), "$500")


if __name__ == "__main__":
    unittest.main()
