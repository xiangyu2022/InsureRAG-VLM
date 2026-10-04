import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_retrieval_before_after import _check_comparable_manifests, _require_retriever
from src.insurerag_vlm.benchmark import _text_retrieval_metrics
from src.insurerag_vlm.qa import compute_retrieval_metrics


class StubPipeline:
    def rank_pages(self, question, data_folder, top_k=10):
        return [{"source": "https://example.org/policy", "page_key": "policy::p0001"}]


class RetrievalEvaluationTests(unittest.TestCase):
    def evaluate(self, record):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.jsonl"
            path.write_text(json.dumps({"question": "What is the deductible?", **record}) + "\n", encoding="utf-8")
            return (
                compute_retrieval_metrics(StubPipeline(), Path(directory), path),
                _text_retrieval_metrics(StubPipeline(), Path(directory), path, 10),
            )

    def test_training_gold_sources_and_url_alias_work_in_both_evaluators(self):
        qa_metrics, benchmark_metrics = self.evaluate({"gold_sources": ["https://example.org/policy#page=1"]})
        self.assertEqual(qa_metrics.recall_at_1, 1.0)
        self.assertEqual(benchmark_metrics["recall_at_1"], 1.0)

    def test_gold_page_keys_work_without_source_labels(self):
        qa_metrics, benchmark_metrics = self.evaluate({"gold_page_keys": ["policy::p0001"]})
        self.assertEqual(qa_metrics.mrr_at_10, 1.0)
        self.assertEqual(benchmark_metrics["mrr_at_10"], 1.0)

    def test_missing_gold_is_rejected_instead_of_scored_as_zero(self):
        with self.assertRaisesRegex(ValueError, "no gold sources"):
            self.evaluate({})

    def test_missing_checkpoint_cannot_be_mislabeled_as_trained_retriever(self):
        _require_retriever("local-hashing")
        with self.assertRaises(FileNotFoundError):
            _require_retriever("models/missing-checkpoint-for-test")

    def test_empty_checkpoint_directory_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                _require_retriever(directory)
            (Path(directory) / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
            _require_retriever(directory)

    def test_implicit_hashing_prefix_cannot_be_labeled_as_a_trained_model(self):
        with self.assertRaisesRegex(ValueError, "prefix selects hashing"):
            _require_retriever("local-bge-trained")

    def test_before_after_reject_different_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before.jsonl", Path(directory) / "after.jsonl"
            before.write_text(json.dumps({"question": "A", "gold_sources": ["policy.pdf#page=1"]}), encoding="utf-8")
            after.write_text(json.dumps({"question": "B", "gold_sources": ["policy.pdf#page=1"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same questions"):
                _check_comparable_manifests(before, after)
            after.write_text(json.dumps({"question": "A", "evidence_sources": ["policy.pdf#page=1"]}), encoding="utf-8")
            _check_comparable_manifests(before, after)

    def test_before_after_reject_different_page_keys_with_same_source(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before.jsonl", Path(directory) / "after.jsonl"
            common = {"question": "A", "gold_sources": ["policy.pdf#page=1"]}
            before.write_text(json.dumps({**common, "gold_page_keys": ["extra::p0001"]}), encoding="utf-8")
            after.write_text(json.dumps({**common, "gold_page_keys": ["extra::p0002"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same questions"):
                _check_comparable_manifests(before, after)


if __name__ == "__main__":
    unittest.main()
