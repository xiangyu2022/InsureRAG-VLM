import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.insurerag_vlm.app import DATA_FOLDER, DemoHandler, _is_personal_policy_query, build_chat_response
from src.insurerag_vlm.vlm import BackendUnavailableError


class AppServingTests(unittest.TestCase):
    def pipeline(self, **result):
        return SimpleNamespace(
            vlm_client=SimpleNamespace(
                backend_label=lambda: "Ollama: qwen3.5:4b", generate_chat=Mock(),
                answer_trace=Mock(return_value={"backend_metadata": {"provider": "ollama", "last_generation": {}}}),
            ),
            query_structured=Mock(return_value={
                "answer": "The policy deductible is $500.", "abstain": False,
                "citations": [{"source": "policy.pdf#page=1"}], "confidence": 0.63,
                "answer_backend": "Ollama: qwen3.5:4b", "generation_used": True,
                "answer_repaired": False, "backend_metadata": {"provider": "ollama"},
                **result,
            }),
        )

    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[])
    def test_document_answer_uses_one_validated_generation_without_forcing_extraction(self, _):
        pipeline = self.pipeline()
        result = build_chat_response("What deductible is in my policy?", pipeline, Path("docs"))
        pipeline.query_structured.assert_called_once_with("What deductible is in my policy?", Path("docs"), top_k=3)
        pipeline.vlm_client.generate_chat.assert_not_called()
        self.assertEqual(result["confidence"], 0.63)
        self.assertTrue(result["generation_used"])
        self.assertEqual(result["backend"], "Ollama: qwen3.5:4b")

    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[SimpleNamespace(term="deductible")])
    def test_document_abstention_cannot_be_overridden_by_glossary(self, _):
        pipeline = self.pipeline(answer="", abstain=True, abstain_reason="Insufficient evidence",
                                 generation_used=False, answer_backend="retrieval-abstention")
        result = build_chat_response("What is my deductible?", pipeline, Path("docs"))
        self.assertTrue(result["abstain"])
        self.assertEqual(result["source"], "abstain")
        self.assertEqual(result["citations"], [])
        self.assertFalse(result["generation_used"])
        pipeline.vlm_client.generate_chat.assert_not_called()

    def test_personal_policy_intent_supports_modifiers_and_coverage_questions(self):
        for question in (
            "What is my collision deductible?", "What is my actual deductible?",
            "What is our current annual premium?", "Do I have collision coverage?",
            "Am I covered for water damage?", "What limit applies to me?",
        ):
            with self.subTest(question=question):
                self.assertTrue(_is_personal_policy_query(question))
        self.assertFalse(_is_personal_policy_query("What does deductible mean?"))
        self.assertFalse(_is_personal_policy_query("My question is: what does deductible mean?"))

    @patch("src.insurerag_vlm.app.search_knowledge")
    def test_public_guides_cannot_answer_personal_amounts_or_call_generation(self, knowledge):
        for question in ("What is my collision deductible?", "What is my actual deductible?"):
            with self.subTest(question=question):
                pipeline = self.pipeline()
                result = build_chat_response(question, pipeline, DATA_FOLDER)
                self.assertTrue(result["abstain"])
                self.assertFalse(result["generation_used"])
                self.assertEqual(result["abstain_reason_code"], "personal_policy_not_available")
                self.assertIn("public reference guides", result["abstain_reason"])
                self.assertEqual(result["backend_metadata"]["last_generation"], {})
                pipeline.query_structured.assert_not_called()
                pipeline.vlm_client.generate_chat.assert_not_called()
                knowledge.assert_not_called()

    @patch("src.insurerag_vlm.app.format_knowledge_answer", return_value="A deductible is...")
    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[SimpleNamespace(term="deductible")])
    def test_modified_personal_question_uses_uploaded_evidence_but_definition_stays_glossary(self, *_):
        pipeline = self.pipeline()
        result = build_chat_response("What is my collision deductible?", pipeline, Path("uploaded"))
        self.assertEqual(result["source"], "document")
        pipeline.query_structured.assert_called_once()
        pipeline.query_structured.reset_mock()
        result = build_chat_response("What does deductible mean?", pipeline, DATA_FOLDER)
        self.assertEqual(result["source"], "knowledge")
        pipeline.query_structured.assert_not_called()

    @patch("src.insurerag_vlm.app.format_knowledge_answer", return_value="A deductible is...")
    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[SimpleNamespace(term="deductible")])
    def test_glossary_is_truthfully_labeled_without_llm_or_document_claim(self, *_):
        pipeline = self.pipeline()
        result = build_chat_response("What is a deductible?", pipeline, Path("docs"))
        pipeline.query_structured.assert_not_called()
        self.assertEqual(result["source"], "knowledge")
        self.assertEqual(result["backend"], "knowledge-base deterministic answer")
        self.assertIsNone(result["confidence"])
        self.assertFalse(result["generation_used"])

    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[])
    def test_repaired_generation_is_exposed(self, _):
        result = build_chat_response("What does this policy cover?", self.pipeline(
            answer_repaired=True, answer_backend="deterministic-evidence-repair"), Path("docs"))
        self.assertTrue(result["answer_repaired"])
        self.assertEqual(result["backend"], "deterministic-evidence-repair")

    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[])
    def test_abstention_displays_plain_language_and_retains_machine_reason(self, _):
        result = build_chat_response("What is my deductible?", self.pipeline(
            answer="", abstain=True, abstain_reason="insufficient_retrieved_evidence"), Path("docs"))
        self.assertIn("available document evidence", result["abstain_reason"])
        self.assertNotIn("insufficient_retrieved_evidence", result["abstain_reason"])
        self.assertEqual(result["abstain_reason_code"], "insufficient_retrieved_evidence")

    def test_active_corpus_distinguishes_public_guides_from_uploaded_files(self):
        class IsolatedHandler(DemoHandler):
            _data_folder = DATA_FOLDER
        self.assertEqual(IsolatedHandler.corpus_status()["label"], "Public reference guides")
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "my_policy.pdf").write_bytes(b"test")
            with patch("src.insurerag_vlm.app.UPLOAD_DIR", folder):
                IsolatedHandler._data_folder = folder
                corpus = IsolatedHandler.corpus_status()
            self.assertEqual(corpus["mode"], "uploaded")
            self.assertEqual(corpus["label"], "my_policy.pdf")
            self.assertEqual(corpus["filenames"], ["my_policy.pdf"])

    @patch("src.insurerag_vlm.app.search_knowledge", return_value=[])
    def test_backend_failure_is_not_replaced_with_a_fake_answer(self, _):
        pipeline = self.pipeline()
        pipeline.query_structured.side_effect = BackendUnavailableError("selected model unavailable")
        with self.assertRaises(BackendUnavailableError):
            build_chat_response("What does this policy cover?", pipeline, Path("docs"))

    def test_api_returns_backend_failure_as_503_json(self):
        handler = object.__new__(DemoHandler)
        handler._handle_get = Mock(side_effect=BackendUnavailableError("selected model unavailable"))
        handler._send_json = Mock()
        handler.do_GET()
        payload = handler._send_json.call_args.args[0]
        self.assertEqual(handler._send_json.call_args.kwargs["status"], 503)
        self.assertFalse(payload["fallback_used"])
        self.assertNotIn("answer", payload)

    @patch("src.insurerag_vlm.app.DocumentRetrievalPipeline")
    def test_uploaded_pdf_uses_document_corpus_instead_of_bundled_curated_data(self, pipeline_type):
        class IsolatedHandler(DemoHandler):
            pass
        IsolatedHandler.use_uploaded_folder(Path("uploaded"), Path("upload-index"))
        config = pipeline_type.call_args.args[0]
        self.assertEqual(config.corpus_source, "documents")
        pipeline_type.return_value.build_index.assert_called_once_with(Path("uploaded"))


if __name__ == "__main__":
    unittest.main()
