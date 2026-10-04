import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


class IndexIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        (self.docs / "policy.txt").write_text("Collision deductible: $500. Liability limit: $100,000.", encoding="utf-8")
        self.pipeline = DocumentRetrievalPipeline(ModelConfig(
            index_dir=self.root / "index", corpus_source="documents", enable_image_signal=False,
            retrieval_model="local-hashing", vlm_model="local-extractive",
        ))
        self.pipeline.build_index(self.docs)

    def test_new_index_describes_encoder_and_loads_with_matching_configuration(self):
        indices = self.pipeline._ensure_indices(self.docs)
        manifest = indices["embedding_manifest"]
        self.assertEqual(manifest["embedding_fingerprint"]["model"], "local-hashing")
        self.assertEqual(manifest["page_dense_shape"], list(indices["page_dense"].shape))
        self.assertEqual(manifest["corpus_root"], str(self.docs.resolve()))
        self.assertTrue(self.pipeline.rank_pages("What is the collision deductible?", self.docs))

    def test_changed_encoder_contract_is_rejected_even_after_index_is_cached(self):
        self.pipeline._ensure_indices(self.docs)
        self.pipeline.retriever.query_instruction = "new query instruction: "
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.pipeline._ensure_indices(self.docs)

    def test_pre_manifest_index_cannot_be_assumed_compatible(self):
        self.pipeline._index_paths()["embedding_manifest"].unlink()
        with self.assertRaisesRegex(ValueError, "identity is missing"):
            self.pipeline._ensure_indices(self.docs)

    def test_wrong_corpus_root_cannot_reuse_existing_index(self):
        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "different corpus"):
            self.pipeline._ensure_indices(other)

    def test_shape_mismatch_is_detected_on_load(self):
        np.save(self.pipeline._index_paths()["page_dense"], np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "array shape"):
            self.pipeline._ensure_indices(self.docs)

    def test_explicit_rebuild_reads_edited_documents_and_replaces_manifest(self):
        path = self.pipeline._index_paths()["embedding_manifest"]
        first = json.loads(path.read_text(encoding="utf-8"))
        self.pipeline._ensure_indices(self.docs)
        (self.docs / "policy.txt").write_text("Collision deductible: $2,000. Liability limit: $500,000.", encoding="utf-8")
        self.pipeline.build_index(self.docs)
        rebuilt = json.loads(path.read_text(encoding="utf-8"))
        self.assertNotEqual(first["corpus_sha256"], rebuilt["corpus_sha256"])
        self.assertIn("$2,000", self.pipeline._ensure_indices(self.docs)["page_meta"][0]["text"])


if __name__ == "__main__":
    unittest.main()
