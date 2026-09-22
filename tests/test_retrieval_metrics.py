import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.insurerag_vlm.retrieval_manifest import scope_and_merge
from src.insurerag_vlm.retrieval_metrics import gold_groups, score_ranking, validate_queries


def example(*sources, question="Which evidence supports this claim?"):
    return {"question": question, "evidence_sources": list(sources), "answerable": True}


class RetrievalMetricTests(unittest.TestCase):
    def test_multi_page_recall_is_not_hit_rate(self):
        scores = score_ranking(example("a", "b"), [{"source": "a"}, {"source": "wrong"}])
        self.assertEqual(scores["recall_at_1"], 0.5)
        self.assertEqual(scores["recall_at_5"], 0.5)
        self.assertEqual(scores["hit_at_5"], 1.0)
        self.assertEqual(scores["mrr_at_10"], 1.0)
        self.assertAlmostEqual(scores["ndcg_at_10"], 1 / (1 + 1 / math.log2(3)))

    def test_ndcg_rewards_second_evidence_and_normalizes(self):
        rows = [{"source": "wrong"}, {"source": "a"}, {"source": "b"}]
        scores = score_ranking(example("a", "b"), rows)
        self.assertEqual(scores["recall_at_5"], 1.0)
        self.assertEqual(scores["mrr_at_10"], 0.5)
        self.assertAlmostEqual(scores["ndcg_at_10"], (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
        self.assertEqual(score_ranking(example("a", "b"), rows[1:])["ndcg_at_10"], 1.0)

    def test_duplicate_prediction_uses_a_rank_without_extra_gain(self):
        scores = score_ranking(example("doc#page=1", "doc#page=2"),
                               [{"source": "doc#page=1"}, {"page_key": "doc::p0001"}, {"source": "doc#page=2"}])
        self.assertEqual(scores["recall_at_5"], 1)
        self.assertAlmostEqual(scores["ndcg_at_10"], 1.5 / (1 + 1 / math.log2(3)))

    def test_aliases_do_not_double_gold_denominator_or_match_other_docs(self):
        row = {**example("https://site/doc#page=1"), "gold_page_keys": ["internal::p0001"]}
        self.assertEqual(len(gold_groups(row)), 1)
        self.assertEqual(score_ranking(row, [{"source": "https://site/doc"}])["recall_at_5"], 1)
        self.assertEqual(score_ranking(row, [{"page_key": "internal::p0001"}])["recall_at_5"], 1)
        self.assertEqual(score_ranking(row, [{"source": "https://other/doc#page=1"}])["recall_at_5"], 0)

    def test_page_number_metadata_does_not_match_page_one(self):
        row = example("doc#page=1")
        self.assertEqual(score_ranking(row, [{"source": "doc", "page_number": 2}])["recall_at_5"], 0)
        row = example("doc#page=2")
        self.assertEqual(score_ranking(row, [{"source": "doc", "page_number": 2}])["recall_at_5"], 1)
        row = {**example("doc"), "gold_page_keys": ["internal::p0002"]}
        self.assertEqual(score_ranking(row, [{"source": "doc#page=1"}])["recall_at_5"], 0)
        self.assertEqual(score_ranking(row, [{"source": "doc#page=2"}])["recall_at_5"], 1)

    def test_single_gold_rank_five_and_six_have_different_recall(self):
        for rank, recall in ((5, 1.0), (6, 0.0)):
            scores = score_ranking(example("gold"), [{"source": "wrong"}] * (rank - 1) + [{"source": "gold"}])
            self.assertEqual(scores["recall_at_5"], recall)
            self.assertEqual(scores["mrr_at_10"], 1 / rank)
            self.assertEqual(scores["ndcg_at_10"], 1 / math.log2(rank + 1))

    def test_model_order_can_differ_between_recall_and_mrr(self):
        from src.insurerag_vlm.retrieval_metrics import mean_scores
        row = example("gold")
        a = mean_scores([score_ranking(row, [{"source": "gold"}]), score_ranking(row, [])])
        b = mean_scores([score_ranking(row, [{"source": "wrong"}] * n + [{"source": "gold"}]) for n in (1, 3)])
        self.assertGreater(b["recall_at_5"], a["recall_at_5"])
        self.assertGreater(a["mrr_at_10"], b["mrr_at_10"])
        self.assertGreater(b["ndcg_at_10"], a["ndcg_at_10"])

    def test_duplicate_gold_and_visual_ids(self):
        self.assertEqual(len(gold_groups(example("doc#page=1", "doc::p0001"))), 1)
        row = {"question": "q", "evidence_page_ids": ["image_1", "image_2"]}
        self.assertEqual(score_ranking(row, [{"page_id": "image_1"}])["recall_at_5"], 0.5)

    def test_rank_ten_included_eleven_excluded(self):
        rows = [{"source": "wrong"}] * 9 + [{"source": "a"}, {"source": "b"}]
        scores = score_ranking(example("a", "b"), rows)
        self.assertEqual(scores["recall_at_5"], 0)
        self.assertEqual(scores["mrr_at_10"], 0.1)
        self.assertAlmostEqual(scores["ndcg_at_10"], (1 / math.log2(11)) / (1 + 1 / math.log2(3)))

    def test_more_than_ten_gold_pages_uses_full_recall_denominator(self):
        sources = [f"p{i}" for i in range(12)]
        scores = score_ranking(example(*sources), [{"source": source} for source in sources])
        self.assertAlmostEqual(scores["recall_at_5"], 5 / 12)
        self.assertEqual(scores["ndcg_at_10"], 1)

    def test_empty_or_misaligned_gold_rejected(self):
        with self.assertRaisesRegex(ValueError, "no gold"):
            score_ranking(example(), [])
        with self.assertRaisesRegex(ValueError, "aligned"):
            gold_groups({**example("a", "b"), "gold_page_keys": ["x"]})

    def test_ambiguous_duplicate_query_is_visible(self):
        rows = [example("a"), example("b")]
        with self.assertWarnsRegex(RuntimeWarning, "duplicate retrieval questions"):
            validate_queries(rows)
        with self.assertRaisesRegex(ValueError, "duplicate retrieval questions"):
            validate_queries(rows, strict=True)


class ManifestTests(unittest.TestCase):
    def test_scope_merges_existing_positives_without_page_or_answer_leakage(self):
        rows = [{**example("policy.pdf#page=2"), "qa_id": "a", "answer": "SECRET"},
                {**example("policy.pdf#page=3"), "qa_id": "b"},
                {**example("other.pdf#page=2"), "qa_id": "c"}]
        scoped = scope_and_merge(rows)
        self.assertEqual(len(scoped), 2)
        self.assertEqual(len(gold_groups(scoped[0])), 2)
        self.assertIn("policy.pdf", scoped[0]["question"])
        self.assertNotIn("#page", scoped[0]["question"])
        self.assertNotIn("SECRET", scoped[0]["question"])
        self.assertEqual(len(scoped[0]["provenance"]), 2)
        validate_queries(scoped, strict=True)

    def test_document_split_and_unique_queries_in_committed_v2(self):
        root = Path(__file__).resolve().parents[1] / "reports/retrieval_eval/v2"
        for name in ("expanded_targeted", "external_official"):
            splits = []
            for split in ("valid", "test"):
                rows = [json.loads(line) for line in (root / name / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertTrue(rows)
                validate_queries(rows, strict=True)
                splits.append({row["source_doc_id"] for row in rows})
            self.assertFalse(splits[0] & splits[1])


class EvaluationEntryPointTests(unittest.TestCase):
    def test_all_entry_points_agree_and_filter_unsupported(self):
        from src.insurerag_vlm.qa import compute_retrieval_metrics
        from src.insurerag_vlm.benchmark import _text_retrieval_metrics
        from src.insurerag_vlm.visual import compute_visual_retrieval_metrics

        class Pipeline:
            def rank_pages(self, question, data_folder, top_k=10):
                return [{"source": "a"}] if question == "q1" else []

        rows = [example("a", "b", question="q1"), example("c", question="q2"),
                {"question": "unsupported", "answerable": False}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qa.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            qa = compute_retrieval_metrics(Pipeline(), Path(directory), path)
            bench = _text_retrieval_metrics(Pipeline(), Path(directory), path, 10)
            with patch("src.insurerag_vlm.visual.visual_search", side_effect=lambda query, **kw: Pipeline().rank_pages(query, directory)):
                visual = compute_visual_retrieval_metrics(path, Path(directory))
            self.assertEqual(qa.recall_at_5, 0.25)  # Macro mean, not pooled page recall.
            self.assertEqual(qa.hit_at_5, 0.5)
            for name, value in qa.__dict__.items():
                self.assertEqual(value, bench[name])
                self.assertEqual(value, visual[name])
            for depth in (3, 5, 9):
                with self.assertRaisesRegex(ValueError, "top_k >= 10"):
                    compute_retrieval_metrics(Pipeline(), Path(directory), path, top_k=depth)
                with self.assertRaises(ValueError):
                    _text_retrieval_metrics(Pipeline(), Path(directory), path, depth)
                with self.assertRaises(ValueError):
                    compute_visual_retrieval_metrics(path, Path(directory), top_k=depth)


if __name__ == "__main__":
    unittest.main()
