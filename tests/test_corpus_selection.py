import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.insurerag_vlm.app import DemoHandler
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


class CorpusSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.curated = self.make_curated("global-curated", 999)

    def make_curated(self, name, amount):
        folder = self.root / name
        folder.mkdir()
        row = {"doc_id": name, "page": 1, "citation": f"{name}.pdf#page=1",
               "text": f"Collision deductible is ${amount}."}
        (folder / "rag_pages.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (folder / "rag_snippets.jsonl").write_text("", encoding="utf-8")
        return folder

    def make_documents(self, name, amount):
        folder = self.root / name
        folder.mkdir()
        (folder / "policy.txt").write_text(f"Collision deductible is ${amount}.", encoding="utf-8")
        return folder

    def pipeline(self, name="index", mode="auto", curated=None):
        return DocumentRetrievalPipeline(ModelConfig(
            index_dir=self.root / name, corpus_source=mode,
            curated_dataset_dir=curated or self.curated,
            enable_image_signal=False, retrieval_model="local-hashing", vlm_model="local-extractive",
        ))

    def test_auto_uses_each_requested_raw_folder_despite_unrelated_global_curated_data(self):
        for name, amount in (("policy-a", 712), ("policy-b", 834)):
            folder = self.make_documents(name, amount)
            pipeline = self.pipeline(name + "-index")
            pipeline.build_index(folder)
            pages = pipeline._ensure_indices(folder)["page_meta"]
            self.assertEqual(len(pages), 1)
            self.assertIn(f"${amount}", pages[0]["text"])
            self.assertNotIn("$999", pages[0]["text"])

    def test_auto_detects_curated_files_in_requested_folder(self):
        requested = self.make_curated("requested", 713)
        pages = self.pipeline()._load_hybrid_corpus(requested)["pages"]
        self.assertEqual(pages[0]["text"], "Collision deductible is $713.")

    def test_explicit_curated_mode_uses_configured_snapshot(self):
        pipeline = self.pipeline(mode="curated")
        corpus = pipeline._load_hybrid_corpus(self.root / "optional-original-pdfs-absent")
        self.assertIn("$999", corpus["pages"][0]["text"])

    def test_missing_auto_folder_fails_instead_of_loading_global_curated_corpus(self):
        with self.assertRaisesRegex(FileNotFoundError, "Requested corpus folder does not exist"):
            self.pipeline().build_index(self.root / "missing")

    def test_partial_curated_folder_has_clear_error(self):
        folder = self.root / "partial"
        folder.mkdir()
        (folder / "rag_pages.jsonl").write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "requires both"):
            self.pipeline()._load_hybrid_corpus(folder)

    def test_cached_index_rejects_edited_source_file(self):
        folder = self.make_documents("policy", 712)
        pipeline = self.pipeline()
        pipeline.build_index(folder)
        pipeline._ensure_indices(folder)
        (folder / "policy.txt").write_text("Collision deductible is $2,500.", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "corpus source, files, or contents changed"):
            pipeline._ensure_indices(folder)

    def test_same_requested_folder_cannot_hide_changed_configured_curated_source(self):
        pipeline = self.pipeline(mode="curated")
        request_folder = self.root / "optional-pdfs"
        pipeline.build_index(request_folder)
        pipeline._ensure_indices(request_folder)
        pipeline.config.curated_dataset_dir = self.make_curated("other-curated", 888)
        with self.assertRaisesRegex(ValueError, "corpus source, files, or contents changed"):
            pipeline._ensure_indices(request_folder)

    @patch("src.insurerag_vlm.app.DocumentRetrievalPipeline")
    def test_demo_explicitly_uses_public_snapshot_when_original_pdfs_are_absent(self, pipeline_type):
        missing = self.root / "public-pdfs-absent"
        class IsolatedHandler(DemoHandler):
            _pipeline = None
            _data_folder = missing
            _index_dir = self.root / "demo-index"
        with patch("src.insurerag_vlm.app.DATA_FOLDER", missing):
            IsolatedHandler.pipeline()
        config = pipeline_type.call_args.args[0]
        self.assertEqual(config.corpus_source, "curated")
        self.assertEqual(config.curated_dataset_dir.name, "04_curated")


if __name__ == "__main__":
    unittest.main()
