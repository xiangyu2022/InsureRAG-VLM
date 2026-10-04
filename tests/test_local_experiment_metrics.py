import unittest

from src.insurerag_vlm.local_answer_metrics import score, aggregate, prompt_ids


class LocalExperimentMetricTests(unittest.TestCase):
    def row(self):
        return {"answerable": True, "evidence": "The deductible is $500.",
                "reference_content": "The deductible is $500.", "source": "a/policy.pdf#page=2"}

    def test_correct_quote_and_source(self):
        metrics = score(self.row(), "The deductible is $500. Source: a/policy.pdf#page=2")
        self.assertEqual(metrics["content_f1"], 1)
        self.assertTrue(metrics["lexical_support_proxy"])

    def test_wrong_number_and_wrong_packet_are_not_supported(self):
        metrics = score(self.row(), "The deductible is $5000. Source: b/policy.pdf#page=2")
        self.assertFalse(metrics["citation_correct"])
        self.assertFalse(metrics["numbers_supported"])
        self.assertFalse(metrics["lexical_support_proxy"])

    def test_abstention_precision_differs_from_recall(self):
        rows = []
        for answerable, text in ((False, "INSUFFICIENT_EVIDENCE"), (True, "INSUFFICIENT_EVIDENCE"),
                                 (False, "The deductible is $500.")):
            row = {**self.row(), "answerable": answerable}
            rows.append({**score(row, text), "answerable": answerable, "seconds": 1, "generated_tokens": 2})
        metrics = aggregate(rows)
        self.assertEqual(metrics["abstention_counts"], {"tp": 1, "fp": 1, "fn": 1})
        self.assertEqual(metrics["abstention_precision"], 0.5)
        self.assertEqual(metrics["abstention_recall"], 0.5)

    def test_retrieved_context_prompt_does_not_reveal_gold_source(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "\n".join(m["content"] for m in messages)

            def __call__(self, text, **kwargs):
                return {"input_ids": text}
        row = {**self.row(), "question": "What is the deductible?",
               "evidence": "SOURCE: retrieved.pdf#page=1\nCoverage evidence.",
               "input_source": "Use SOURCE identifiers from the evidence."}
        rendered = prompt_ids(Tokenizer(), row)
        self.assertNotIn(row["source"], rendered)
        self.assertIn("retrieved.pdf#page=1", rendered)
        self.assertNotIn("Source: Use SOURCE", rendered)
        self.assertNotIn("followed by the supplied source exactly", rendered)
        self.assertIn("one SOURCE identifier", rendered)


if __name__ == "__main__":
    unittest.main()
