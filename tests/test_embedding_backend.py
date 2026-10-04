import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.insurerag_vlm.retriever import EmbeddingBackendError, EmbeddingRetriever


class EmbeddingBackendTests(unittest.TestCase):
    def test_empty_snippet_index_returns_no_candidates_without_embedding(self):
        retriever = EmbeddingRetriever("local-hashing")
        with patch.object(retriever, "embed_texts") as embed:
            self.assertEqual(retriever.search("query", np.zeros((0, 0))), [])
            self.assertEqual(retriever.search("query", np.zeros((0, 512)), return_scores=True), [])
            embed.assert_not_called()

    def test_incompatible_vector_dimensions_fail_with_rebuild_guidance(self):
        retriever = EmbeddingRetriever("local-hashing")
        with self.assertRaisesRegex(EmbeddingBackendError, "rebuild the index"):
            retriever.search("query", np.ones((2, 384)))

    @patch.dict("os.environ", {}, clear=True)
    def test_only_explicit_hashing_uses_baseline(self):
        retriever = EmbeddingRetriever("local-hashing")
        vectors = retriever.embed_texts(["policy deductible", "policy limit"])
        self.assertEqual(vectors.shape, (2, 512))
        self.assertTrue(retriever.backend_metadata()["is_baseline"])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
        with self.assertRaises(EmbeddingBackendError):
            EmbeddingRetriever("local-pretend-bge").embed_texts(["query"])

    @patch.dict("os.environ", {}, clear=True)
    def test_requested_hf_model_without_credentials_does_not_fall_back(self):
        retriever = EmbeddingRetriever("BAAI/bge-base-en-v1.5")
        with patch.object(retriever, "_local_hash_embeddings") as hashing:
            with self.assertRaisesRegex(EmbeddingBackendError, "HF_API_TOKEN"):
                retriever.embed_texts(["query"])
            hashing.assert_not_called()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-key", "HF_API_TOKEN": "test-only-token"}, clear=True)
    def test_missing_local_checkpoint_cannot_turn_into_hosted_request(self):
        retriever = EmbeddingRetriever("models/missing-checkpoint", use_hf_api=False)
        with patch.object(retriever, "_openai_embeddings") as hosted:
            with self.assertRaisesRegex(EmbeddingBackendError, "unavailable"):
                retriever.embed_texts(["query"])
            hosted.assert_not_called()

    @patch.dict("os.environ", {"HF_API_TOKEN": "test-only-token"}, clear=True)
    def test_openai_selection_does_not_fall_back_to_huggingface(self):
        with self.assertRaisesRegex(EmbeddingBackendError, "OPENAI_API_KEY"):
            EmbeddingRetriever("text-embedding-3-small", use_hf_api=False).embed_texts(["query"])

    @patch.dict("os.environ", {}, clear=True)
    def test_local_model_directory_requires_config_and_keeps_transformer_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            retriever = EmbeddingRetriever(directory, pooling="mean", max_length=128)
            with self.assertRaisesRegex(EmbeddingBackendError, "config.json"):
                retriever.embed_texts(["query"])
            (Path(directory) / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
            with patch.object(retriever, "_local_transformer_embeddings", return_value=np.ones((1, 3))) as encoder:
                self.assertEqual(retriever.embed_texts(["query"]).shape, (1, 3))
                encoder.assert_called_once()
            self.assertEqual(retriever.backend_metadata()["backend"], "local-transformer")

    @patch.dict("os.environ", {}, clear=True)
    def test_checkpoint_declared_cls_pooling_is_used_without_name_guessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"hidden_size": 384, "max_position_embeddings": 512}', encoding="utf-8")
            (root / "1_Pooling").mkdir()
            (root / "1_Pooling" / "config.json").write_text(
                '{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}', encoding="utf-8")
            metadata = EmbeddingRetriever(directory).backend_metadata()
            self.assertEqual(metadata["pooling"], "cls")
            self.assertEqual(metadata["pooling_source"], "1_Pooling/config.json")
            self.assertEqual(metadata["max_length"], 512)
            self.assertEqual(metadata["embedding_dim"], 384)

    @patch.dict("os.environ", {}, clear=True)
    def test_unknown_checkpoint_pooling_requires_an_explicit_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"max_position_embeddings": 512}', encoding="utf-8")
            with self.assertRaisesRegex(EmbeddingBackendError, "does not declare supported pooling"):
                EmbeddingRetriever(directory).index_fingerprint()
            (root / "insurerag_embedding_config.json").write_text(
                '{"pooling": "mean", "max_length": 384, "normalize": true}', encoding="utf-8")
            self.assertEqual(EmbeddingRetriever(directory).index_fingerprint()["pooling"], "mean")

    @patch.dict("os.environ", {}, clear=True)
    def test_fingerprint_tracks_checkpoint_content_and_encoding_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"hidden_size": 3, "max_position_embeddings": 512}', encoding="utf-8")
            weights = root / "model.safetensors"
            weights.write_bytes(b"fixture weights v1")
            retriever = EmbeddingRetriever(directory, pooling="cls")
            first = retriever.index_fingerprint()
            self.assertEqual(first, retriever.index_fingerprint())
            weights.write_bytes(b"fixture weights version2")
            self.assertNotEqual(first["checkpoint_sha256"], retriever.index_fingerprint()["checkpoint_sha256"])
            self.assertNotEqual(retriever.index_fingerprint(), EmbeddingRetriever(directory, pooling="mean").index_fingerprint())
            self.assertNotEqual(retriever.index_fingerprint(), EmbeddingRetriever(directory, pooling="cls", query_instruction="query: ").index_fingerprint())

    @patch.dict("os.environ", {}, clear=True)
    def test_query_instruction_applies_to_queries_only(self):
        retriever = EmbeddingRetriever("local-hashing", query_instruction="query: ")
        with patch.object(retriever, "embed_texts", return_value=np.asarray([[1.0, 0.0]])) as embed:
            retriever.search("deductible", np.asarray([[1.0, 0.0]]))
            embed.assert_called_once_with(["query: deductible"])
        with patch.object(retriever, "_local_hash_embeddings", return_value=np.asarray([[1.0, 0.0]])) as embed:
            retriever.embed_texts(["policy passage"])
            embed.assert_called_once_with(["policy passage"])

    @patch.dict("os.environ", {}, clear=True)
    def test_ambiguous_concatenated_pooling_is_not_silently_reinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"max_position_embeddings": 512}', encoding="utf-8")
            (root / "1_Pooling").mkdir()
            (root / "1_Pooling" / "config.json").write_text(
                '{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true}', encoding="utf-8")
            with self.assertRaisesRegex(EmbeddingBackendError, "ambiguous pooling"):
                EmbeddingRetriever(directory).backend_metadata()


if __name__ == "__main__":
    unittest.main()
