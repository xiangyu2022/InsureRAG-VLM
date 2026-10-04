import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import eval_sft_adapter
from src.insurerag_vlm.sft import QwenLoraSFTConfig, _SFTProgressCallback, read_sft_records, run_lora_sft
from src.insurerag_vlm.sft_integrity import (
    PROVENANCE_FILENAME,
    audit_evaluation_records,
    build_training_provenance,
    load_provenance,
    write_provenance,
)


def record(record_id="train-1", source="policy.pdf#page=1", answerable=True):
    return {
        "record_id": record_id,
        "question": "What is the deductible?",
        "evidence": "The deductible is $500.",
        "answer": "$500" if answerable else "Insufficient evidence.",
        "source": source,
        "answerable": answerable,
    }


class SFTIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / "train.jsonl"
        self.write_dataset([record()])

    def write_dataset(self, rows, path=None):
        path = path or self.dataset
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def manifest(self, rows=None, parents=()):
        return build_training_provenance(rows or [record()], self.dataset, "base-model", parents)

    def test_changed_ids_and_answers_do_not_hide_prompt_overlap(self):
        copied = record("renamed")
        copied["answer"] = "Different annotation"
        audit = audit_evaluation_records([copied], self.manifest())
        self.assertEqual(audit["overlap_counts"]["record_ids"], 0)
        self.assertEqual(audit["overlap_counts"]["prompt_fingerprints"], 1)
        with self.assertRaisesRegex(ValueError, "Held-out evaluation rejected"):
            audit_evaluation_records([copied], self.manifest(), require_heldout=True)

    def test_other_page_of_same_document_is_not_document_heldout(self):
        heldout = record("test-1", " POLICY.PDF#page=9 ")
        heldout["question"] = "Which limits apply?"
        audit = audit_evaluation_records([heldout], self.manifest())
        self.assertEqual(audit["overlap_counts"]["source_fingerprints"], 0)
        self.assertEqual(audit["overlap_counts"]["document_fingerprints"], 1)
        self.assertFalse(audit["document_disjoint_sft_check_passed"])

    def test_same_template_on_unseen_document_can_pass(self):
        audit = audit_evaluation_records(
            [record("test-1", "unseen-policy.pdf#page=2")], self.manifest(), require_heldout=True
        )
        self.assertTrue(audit["document_disjoint_sft_check_passed"])
        self.assertEqual(audit["overlap_counts"]["question_fingerprints"], 1)

    def test_retrieval_context_source_exposure_is_recorded(self):
        training = record()
        training["retrieval_context_sources"] = ["other-policy.pdf#page=2"]
        audit = audit_evaluation_records(
            [record("test-1", "other-policy.pdf#page=5")], self.manifest([training])
        )
        self.assertEqual(audit["overlap_counts"]["document_fingerprints"], 1)

    def test_continued_adapter_retains_ancestral_training_exposure(self):
        parent = self.root / "parent"
        write_provenance(parent, self.manifest())
        continued = self.manifest([record("train-2", "second-policy.pdf#page=1")], [parent])
        self.assertTrue(continued["provenance_complete"])
        self.assertEqual(set(continued["record_ids"]), {"train-1", "train-2"})
        with self.assertRaisesRegex(ValueError, "document_fingerprints overlap"):
            audit_evaluation_records([record("test-1", "policy.pdf#page=8")], continued, True)

    def test_missing_lineage_cannot_become_clean_after_continuation(self):
        unknown_parent = self.root / "legacy-adapter"
        continued = self.manifest(parents=[unknown_parent])
        self.assertFalse(continued["provenance_complete"])
        parent = self.root / "continued"
        write_provenance(parent, continued)
        second_continuation = self.manifest(parents=[parent])
        with self.assertRaisesRegex(ValueError, "missing or incomplete"):
            audit_evaluation_records([record("new", "unseen.pdf")], second_continuation, True)
        self.assertFalse(audit_evaluation_records([record()], None)["provenance_complete"])

    def test_missing_sources_prevent_document_disjoint_claims(self):
        with self.assertRaisesRegex(ValueError, "source identifiers missing"):
            audit_evaluation_records([record("new", "")], self.manifest(), True)

    def test_checkpoint_receives_same_cumulative_manifest(self):
        manifest = self.manifest()
        callback = _SFTProgressCallback(self.root, manifest)
        callback.on_save(None, SimpleNamespace(global_step=25, max_steps=100, epoch=0.25), None)
        checkpoint = self.root / "checkpoint-25"
        self.assertTrue((checkpoint / PROVENANCE_FILENAME).exists())
        self.assertEqual(load_provenance(checkpoint), manifest)

    def test_training_rejects_explicit_eval_rows_before_loading_cuda(self):
        eval_row = record()
        eval_row["split"] = "test"
        self.write_dataset([eval_row])
        with patch("src.insurerag_vlm.sft._require_cuda") as cuda:
            with self.assertRaisesRegex(ValueError, "non-training splits"):
                run_lora_sft(QwenLoraSFTConfig(dataset_path=self.dataset))
            cuda.assert_not_called()

    def test_read_zero_limit_is_rejected_instead_of_reading_one_row(self):
        with self.assertRaisesRegex(ValueError, "max_samples must be positive"):
            read_sft_records(self.dataset, max_samples=0)


class AdapterMetricTests(unittest.TestCase):
    def test_zero_requested_stratum_selects_no_records_from_it(self):
        rows = [record("a"), record("u", answerable=False)]
        self.assertEqual([row["record_id"] for row in eval_sft_adapter._pick_records(rows, 0, 1)], ["u"])
        self.assertEqual([row["record_id"] for row in eval_sft_adapter._pick_records(rows, 1, 0)], ["a"])
        self.assertEqual(eval_sft_adapter._pick_records(rows, 0, 0), [])
        with self.assertRaises(ValueError):
            eval_sft_adapter._pick_records(rows, -1, 1)

    def test_empty_metrics_are_null_and_four_successes_have_uncertainty(self):
        empty = eval_sft_adapter._summarize([])
        self.assertIsNone(empty["overall_f1"])
        self.assertIsNone(empty["unsupported_prediction_abstain_rate"])
        self.assertIsNone(empty["unsupported_prediction_abstain_wilson_95"])
        row = {"answerable": False, "exact_match": 1, "f1": 1, "reference_abstains": True, "prediction_abstains": True}
        summary = eval_sft_adapter._summarize([row] * 4)
        self.assertEqual(summary["unsupported_prediction_abstain_count"], 4)
        self.assertEqual(summary["unsupported_count"], 4)
        self.assertAlmostEqual(summary["unsupported_prediction_abstain_wilson_95"][0], 0.510109, places=5)
        self.assertIsNone(summary["answerable_f1"])

    def test_cli_writes_honest_diagnostic_without_gpu_and_creates_markdown_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "eval.jsonl"
            dataset.write_text(json.dumps(record()) + "\n", encoding="utf-8")
            json_path, md_path = root / "json" / "result.json", root / "markdown" / "result.md"
            argv = ["eval_sft_adapter.py", "--dataset-path", str(dataset), "--all-records",
                    "--adapter-dir", str(root / "legacy"), "--output-json", str(json_path), "--output-md", str(md_path)]
            generated = [{**record(), "reference": "$500", "prediction": "$500", "exact_match": 1, "f1": 1,
                          "reference_abstains": False, "prediction_abstains": False}]
            with patch("sys.argv", argv), patch.object(eval_sft_adapter, "_generate_for_records", return_value=generated):
                eval_sft_adapter.main()
            result = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(result["evaluation_role"], "diagnostic_only_not_a_heldout_benchmark")
            self.assertFalse(result["provenance_audit"]["provenance_complete"])
            self.assertIsNone(result["summary"]["adapter"]["unsupported_prediction_abstain_rate"])
            self.assertIn("N/A", md_path.read_text(encoding="utf-8"))

    def test_heldout_cli_requires_explicit_dataset_before_generation(self):
        with patch("sys.argv", ["eval_sft_adapter.py", "--heldout-evaluation"]), \
                patch.object(eval_sft_adapter, "_generate_for_records") as generate:
            with self.assertRaises(SystemExit) as error:
                eval_sft_adapter.main()
            self.assertEqual(error.exception.code, 2)
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
