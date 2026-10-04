import os
import unittest
import tempfile
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.vlm import BackendConfigurationError, BackendUnavailableError, VLMClient


def response(payload):
    result = Mock()
    result.json.return_value = payload
    result.raise_for_status.return_value = None
    return result


def tags(model="qwen3.5:4b", digest="sha256:test-model"):
    return {"models": [
        {"name": "unrelated:latest", "digest": "sha256:other-model"},
        {"name": model, "model": model, "digest": digest,
         "capabilities": ["completion", "vision", "thinking"],
         "details": {"quantization_level": "Q4_K_M", "parameter_size": "4.66B"}},
    ]}


def backend_get(url, **kwargs):
    if url.endswith("/api/tags"):
        return response(tags())
    if url.endswith("/api/version"):
        return response({"version": "test-ollama"})
    raise AssertionError(f"Unexpected URL: {url}")


def chat_response(content="The collision deductible is $500.\n\nSOURCE: policy.pdf#page=1", **extra):
    return response({"model": "qwen3.5:4b", "message": {"content": content}, "done": True,
                     "done_reason": "stop", "eval_count": 24, "prompt_eval_count": 90, **extra})


class ExplicitBackendTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_api_credentials_and_installed_models_do_not_activate_generation(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key", "HF_API_TOKEN": "test-key", "OLLAMA_MODEL": "unrelated:latest"}), \
                patch("src.insurerag_vlm.vlm.requests.get") as get, patch("src.insurerag_vlm.vlm.requests.post") as post:
            config = ModelConfig()
            self.assertEqual(config.vlm_model, "local-extractive")
            client = VLMClient(config.vlm_model)
            self.assertFalse(client.is_real_llm())
            self.assertIn("$500", client.generate("Context:\nSOURCE: policy.pdf#page=1\nCollision deductible is $500.\n\nQuestion:\nWhat is the collision deductible?\n\nAnswer:"))
            get.assert_not_called()
            post.assert_not_called()

    def test_model_config_uses_only_explicit_insurerag_model_environment(self):
        with patch.dict(os.environ, {"INSURERAG_VLM_MODEL": "ollama:qwen3.5:4b", "OPENAI_EMBEDDING_MODEL": "paid-embedding-model"}):
            self.assertEqual(ModelConfig().vlm_model, "ollama:qwen3.5:4b")
            self.assertEqual(ModelConfig().retrieval_model, "local-hashing")

    def test_ambiguous_model_and_conflicting_provider_are_rejected(self):
        for name, provider in [("some-model", None), ("Qwen/Qwen3.5-4B", None), ("local-extractive", "openai"), ("ollama:qwen3.5:4b", "openai")]:
            with self.subTest(name=name, provider=provider), self.assertRaises(BackendConfigurationError):
                VLMClient(name, provider=provider)

    def test_explicit_hosted_provider_is_not_overridden_by_other_keys(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key", "HF_API_TOKEN": "test-key"}), \
                patch("src.insurerag_vlm.vlm.requests.get") as get:
            client = VLMClient("openai:gpt-4o-mini")
            with patch.object(client, "_call_openai_chat", return_value="openai answer") as selected, \
                    patch.object(client, "_call_anthropic_chat") as other:
                self.assertEqual(client.generate_chat("system", "user"), "openai answer")
                selected.assert_called_once_with("system", "user")
                other.assert_not_called()
            self.assertEqual(client.model_name, "gpt-4o-mini")
            get.assert_not_called()

    def test_hosted_provider_missing_credential_does_not_fall_back(self):
        with self.assertRaisesRegex(BackendConfigurationError, "credential"):
            VLMClient("openai:gpt-4o-mini")
        with self.assertRaisesRegex(BackendConfigurationError, "use_hf_api"):
            VLMClient("hf:organization/model", hf_api_token="test-key", use_hf_api=False)

    def test_exact_missing_ollama_tag_fails_without_any_generation(self):
        with patch("src.insurerag_vlm.vlm.requests.get", return_value=response(tags(model="qwen3.5:9b"))), \
                patch("src.insurerag_vlm.vlm.requests.post") as post:
            with self.assertRaisesRegex(BackendUnavailableError, "qwen3.5:4b.*not installed"):
                VLMClient("ollama:qwen3.5:4b", openai_api_key="test-key")
            post.assert_not_called()

    def test_missing_server_and_missing_digest_fail_closed(self):
        with patch("src.insurerag_vlm.vlm.requests.get", side_effect=requests.ConnectionError("offline")):
            with self.assertRaisesRegex(BackendUnavailableError, "Cannot reach"):
                VLMClient("qwen3.5:4b")
        with patch("src.insurerag_vlm.vlm.requests.get", return_value=response(tags(digest=None))):
            with self.assertRaisesRegex(BackendUnavailableError, "digest"):
                VLMClient("qwen3.5:4b")

    def test_disabled_ollama_conflicts_with_explicit_selection(self):
        with patch.dict(os.environ, {"INSURERAG_USE_OLLAMA": "0"}):
            with self.assertRaisesRegex(BackendConfigurationError, "disables"):
                VLMClient("ollama:qwen3.5:4b")


class OllamaGenerationTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        mock_get = patch("src.insurerag_vlm.vlm.requests.get", side_effect=backend_get)
        self.get = mock_get.start()
        self.addCleanup(mock_get.stop)

    def test_generation_sends_exact_model_and_api_non_thinking_flag(self):
        client = VLMClient("ollama:qwen3.5:4b", ollama_base_url="http://localhost:11435",
                           generation_options={"num_predict": 96, "seed": 17}, expected_model_digest="sha256:test-model")
        with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response()) as post:
            answer = client.generate_chat("ground only in supplied evidence", "deductible?")
        self.assertIn("$500", answer)
        url = post.call_args.args[0]
        body = post.call_args.kwargs["json"]
        self.assertEqual(url, "http://localhost:11435/api/chat")
        self.assertEqual(body["model"], "qwen3.5:4b")
        self.assertIs(body["think"], False)
        self.assertNotIn("think", body["options"])
        self.assertEqual(body["options"]["seed"], 17)
        self.assertEqual(body["options"]["num_predict"], 96)
        self.assertFalse(body["stream"])
        metadata = client.backend_metadata()
        self.assertEqual(metadata["model_digest"], "sha256:test-model")
        self.assertEqual(metadata["ollama_version"], "test-ollama")
        self.assertEqual(metadata["last_generation"]["eval_count"], 24)
        self.assertNotIn("test-key", str(metadata))

    def test_response_schema_is_sent_as_api_format_without_postprocessing_json(self):
        client = VLMClient("qwen3.5:4b")
        schema = {"type": "object", "properties": {"amount": {"type": "string"}}, "required": ["amount"], "additionalProperties": False}
        raw = '```json\n{"amount":"$500"}\n```'
        with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response(raw)) as post:
            answer = client.generate_chat("system", "user", response_format=schema)
        self.assertEqual(post.call_args.kwargs["json"]["format"], schema)
        self.assertEqual(answer, raw)  # A strict-JSON evaluator must still reject actual fences.
        self.assertEqual(client.backend_metadata()["last_generation"]["response_format"], schema)
        with patch("src.insurerag_vlm.vlm.requests.post") as post:
            with self.assertRaisesRegex(BackendConfigurationError, "response_format"):
                VLMClient("local-extractive").generate_chat("system", "user", response_format="json")
            with self.assertRaises(BackendConfigurationError):
                client.generate_chat("system", "user", response_format="xml")
            post.assert_not_called()

    def test_mutated_model_digest_and_unexpected_response_model_are_rejected(self):
        client = VLMClient("qwen3.5:4b")
        self.get.side_effect = lambda *args, **kwargs: response(tags(digest="sha256:changed"))
        with patch("src.insurerag_vlm.vlm.requests.post") as post:
            with self.assertRaisesRegex(BackendUnavailableError, "digest changed"):
                client.generate("prompt")
            post.assert_not_called()
        self.get.side_effect = backend_get
        with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response(model="unrelated:latest")):
            with self.assertRaisesRegex(BackendUnavailableError, "unexpected model"):
                client.generate("prompt")

    def test_digest_pin_detects_different_installed_weights(self):
        with self.assertRaisesRegex(BackendConfigurationError, "digest does not match"):
            VLMClient("qwen3.5:4b", expected_model_digest="different-digest")

    def test_generation_transport_failure_never_uses_extractive_or_paid_fallback(self):
        client = VLMClient("qwen3.5:4b", openai_api_key="test-key")
        with patch("src.insurerag_vlm.vlm.requests.post", side_effect=requests.Timeout()), \
                patch.object(client, "_local_extractive_answer") as extractive, \
                patch.object(client, "_call_openai_chat") as hosted:
            with self.assertRaisesRegex(BackendUnavailableError, "no fallback"):
                client.generate("prompt")
            extractive.assert_not_called()
            hosted.assert_not_called()

    def test_no_final_answer_raises_and_truncation_is_visible(self):
        client = VLMClient("qwen3.5:4b")
        with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response("", done_reason="length")):
            with self.assertRaisesRegex(BackendUnavailableError, "no usable final answer"):
                client.generate("prompt")
        self.assertTrue(client.backend_metadata()["last_generation"]["truncated"])
        with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response("<think>unfinished")):
            with self.assertRaises(BackendUnavailableError):
                client.generate("prompt")

    def test_extractive_and_retrieval_gate_do_not_reuse_llm_metadata(self):
        client = VLMClient("qwen3.5:4b")
        client.last_generation_metadata = {"eval_count": 100}
        no_evidence = client.answer_trace(invoked=False)
        self.assertFalse(no_evidence["generation_used"])
        self.assertEqual(no_evidence["answer_backend"], "retrieval-abstention")
        self.assertEqual(no_evidence["backend_metadata"]["last_generation"], {})
        extractive = client.answer_trace(invoked=True, force_extractive=True)
        self.assertEqual(extractive["answer_backend"], "local-extractive")
        self.assertEqual(extractive["backend_metadata"]["last_generation"], {})

    def test_metadata_is_a_copy_and_isolated_between_request_threads(self):
        client = VLMClient("qwen3.5:4b")
        client.last_generation_metadata = {"eval_count": 100}
        metadata = client.backend_metadata()
        metadata["generation_options"]["seed"] = -1
        self.assertEqual(client.generation_options["seed"], 42)
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(lambda: client.last_generation_metadata).result(), {})
        self.assertEqual(client.last_generation_metadata["eval_count"], 100)

    def test_pipeline_constructor_passes_explicit_runtime_settings_and_empty_gate_trace(self):
        config = ModelConfig(vlm_model="ollama:qwen3.5:4b", ollama_base_url="http://localhost:11435",
                             ollama_generation_options={"num_predict": 72}, vlm_expected_digest="sha256:test-model")
        pipeline = DocumentRetrievalPipeline(config)
        self.assertEqual(pipeline.vlm_client.generation_options["num_predict"], 72)
        with patch.object(pipeline, "merge_candidates", return_value=[]), \
                patch("src.insurerag_vlm.vlm.requests.post") as post:
            result = pipeline.query_structured("What is the deductible?", Path("unused"))
        self.assertFalse(result["generation_used"])
        self.assertTrue(result["abstain"])
        self.assertEqual(result["answer_backend"], "retrieval-abstention")
        post.assert_not_called()

    def test_image_qa_sends_local_bytes_and_records_digest_without_changing_text_mode(self):
        client = VLMClient("qwen3.5:4b")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "declarations.png"
            Image.new("RGB", (20, 30), color="white").save(path)
            content = path.read_bytes()
            with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response()) as post:
                client.generate_with_images("Read the collision deductible in this image.", [path], response_format="json")
            body = post.call_args.kwargs["json"]
            self.assertEqual(base64.b64decode(body["messages"][1]["images"][0]), content)
            self.assertFalse(body["think"])
            self.assertEqual(body["format"], "json")
            metadata = client.backend_metadata()["last_generation"]
            self.assertEqual(metadata["input_modality"], "text+image")
            self.assertEqual(metadata["image_count"], 1)
            self.assertEqual(metadata["images"][0]["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(metadata["images"][0]["width"], 20)
            with patch("src.insurerag_vlm.vlm.requests.post", return_value=chat_response()) as post:
                client.generate("normal text prompt")
            self.assertNotIn("images", post.call_args.kwargs["json"]["messages"][1])
            self.assertEqual(client.backend_metadata()["last_generation"]["image_count"], 0)

    def test_image_qa_rejects_unsupported_backend_or_model_without_discarding_images(self):
        with patch("src.insurerag_vlm.vlm.requests.post") as post:
            with self.assertRaisesRegex(BackendConfigurationError, "Ollama vision"):
                VLMClient("local-extractive").generate_with_images("prompt", [Path("unused.png")])
            client = VLMClient("qwen3.5:4b")
            client._model_info["capabilities"] = ["completion"]
            with self.assertRaisesRegex(BackendConfigurationError, "vision support"):
                client.generate_with_images("prompt", [Path("unused.png")])
            post.assert_not_called()

    def test_image_qa_rejects_corrupt_local_file_and_bad_count_before_generation(self):
        client = VLMClient("qwen3.5:4b")
        with tempfile.TemporaryDirectory() as folder, patch("src.insurerag_vlm.vlm.requests.post") as post:
            path = Path(folder) / "broken.png"
            path.write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(BackendConfigurationError, "Cannot use image"):
                client.generate_with_images("prompt", [path])
            with self.assertRaisesRegex(BackendConfigurationError, "one and four"):
                client.generate_with_images("prompt", [])
            post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
