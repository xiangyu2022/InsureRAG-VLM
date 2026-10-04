import unittest

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.query_understanding import understand_query


class ContextPackingTests(unittest.TestCase):
    def pipeline(self, budget=800, page_budget=5000):
        return DocumentRetrievalPipeline(ModelConfig(
            vlm_model="local-extractive", retrieval_model="local-hashing",
            max_context_chars=budget, max_page_chars=page_budget,
        ))

    def test_first_ranked_evidence_survives_long_lower_ranked_declarations_page(self):
        pages = [
            {"source": "guide.pdf#page=4", "document_type": "base_policy", "score": 0.95,
             "text_snippet": "Collision deductible is $712.", "snippet_support": ["Collision deductible is $712."]},
            {"source": "guide.pdf#page=1", "document_type": "declarations", "score": 0.8,
             "text_snippet": "Unrelated premium narrative. " * 500},
        ]
        context = self.pipeline().pack_long_context(pages, 2, understand_query("What is the collision deductible?"))
        self.assertIn("Collision deductible is $712.", context)
        self.assertLess(context.index("SOURCE: guide.pdf#page=4"), context.index("SOURCE: guide.pdf#page=1"))
        self.assertLessEqual(len(context), 800)

    def test_oversized_first_page_does_not_evict_other_selected_page(self):
        context = self.pipeline(budget=500).pack_long_context([
            {"source": "first.pdf#page=1", "text_snippet": "First page evidence. " * 500},
            {"source": "second.pdf#page=2", "text_snippet": "Second page contains the deductible: $900."},
        ], 2)
        self.assertIn("SOURCE: first.pdf#page=1\n", context)
        self.assertIn("SOURCE: second.pdf#page=2\n", context)
        self.assertIn("Second page contains the deductible: $900.", context)
        self.assertLessEqual(len(context), 500)

    def test_duplicate_and_containing_snippets_do_not_inflate_context(self):
        page = {"source": "policy.pdf#page=1", "text_snippet": "The deductible is $500."}
        duplicate = {**page, "snippet_support": [
            "The deductible is $500.", "The   deductible is $500.",
            "Before the clause. The deductible is $500. After the clause.",
        ]}
        pipeline = self.pipeline()
        self.assertEqual(pipeline.pack_long_context([page], 1), pipeline.pack_long_context([duplicate], 1))

    def test_small_budgets_never_emit_partial_source_headers(self):
        pages = [
            {"source": "a.pdf#page=1", "text_snippet": "Short evidence."},
            {"source": "b.pdf#page=2", "text_snippet": "Other evidence."},
        ]
        valid_headers = {"SOURCE: a.pdf#page=1", "SOURCE: b.pdf#page=2"}
        for budget in (0, 1, 10, 25, 60, 100, 150, 200):
            with self.subTest(budget=budget):
                context = self.pipeline(budget=budget).pack_long_context(pages, 2)
                self.assertLessEqual(len(context), budget)
                self.assertTrue(all(line in valid_headers for line in context.splitlines() if line.startswith("SOURCE")))
                if context:
                    self.assertTrue(context.startswith("SOURCE: a.pdf#page=1\n"))

    def test_repeated_page_source_is_packed_once(self):
        page = {"source": "policy.pdf#page=1", "text_snippet": "Relevant evidence."}
        context = self.pipeline().pack_long_context([page, page], 2)
        self.assertEqual(context.count("SOURCE:"), 1)


if __name__ == "__main__":
    unittest.main()
