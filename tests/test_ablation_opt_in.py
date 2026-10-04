import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.insurerag_vlm.ablation import run_ablation


class AblationOptInTests(unittest.TestCase):
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-key"}, clear=True)
    def test_api_key_alone_does_not_run_hosted_ablation(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("src.insurerag_vlm.ablation.DocumentRetrievalPipeline") as pipeline_type, \
             patch("src.insurerag_vlm.ablation.compute_retrieval_metrics", return_value=SimpleNamespace(evaluated_count=0)), \
             patch("src.insurerag_vlm.ablation._answer_metrics", return_value=({}, [])), \
             patch("src.insurerag_vlm.ablation._write_summary"):
            root = Path(directory)
            run_ablation(root, root / "qa.jsonl", root / "output", index_dir=root / "index", visual_index_dir=root / "visual")
            self.assertEqual(pipeline_type.call_count, 1)
            self.assertEqual(pipeline_type.call_args.args[0].vlm_model, "local-extractive")

    @patch.dict("os.environ", {}, clear=True)
    def test_explicit_hosted_run_without_credentials_fails_before_evaluation(self):
        with patch("src.insurerag_vlm.ablation.DocumentRetrievalPipeline") as pipeline_type:
            with self.assertRaisesRegex(ValueError, "--include-openai requires"):
                run_ablation(Path("docs"), Path("qa.jsonl"), Path("output"), include_openai=True)
            pipeline_type.assert_not_called()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-only-key"}, clear=True)
    def test_explicit_hosted_opt_in_runs_separate_openai_backend(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("src.insurerag_vlm.ablation.DocumentRetrievalPipeline") as pipeline_type, \
             patch("src.insurerag_vlm.ablation.compute_retrieval_metrics", return_value=SimpleNamespace(evaluated_count=0)), \
             patch("src.insurerag_vlm.ablation._answer_metrics", return_value=({}, [])), \
             patch("src.insurerag_vlm.ablation._write_summary"):
            root = Path(directory)
            run_ablation(root, root / "qa.jsonl", root / "output", index_dir=root / "index",
                         visual_index_dir=root / "visual", include_openai=True)
            self.assertEqual(pipeline_type.call_count, 2)
            config = pipeline_type.call_args.args[0]
            self.assertFalse(config.use_hf_api)
            self.assertEqual(config.retrieval_model, "text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()
