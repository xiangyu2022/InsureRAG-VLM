import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.query_understanding import understand_query
from src.insurerag_vlm.tables import build_table_records
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.visual import build_lightweight_page_image_embeddings


class LightweightImageEmbeddingTests(unittest.TestCase):
    def test_embedding_builder_handles_real_and_missing_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "page.png"
            image = Image.new("RGB", (128, 192), color="white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 8, 120, 48), outline="black", width=2)
            draw.text((12, 16), "DECLARATIONS", fill="black")
            image.save(image_path)

            embeddings, pages = build_lightweight_page_image_embeddings(
                [
                    {"source": "doc.pdf#page=1", "image_path": str(image_path)},
                    {"source": "doc.pdf#page=2", "image_path": str(Path(tmpdir) / "missing.png")},
                ]
            )

            self.assertEqual(embeddings.shape, (2, 32))
            self.assertEqual(pages[0]["image_stats"]["feature_dim"], 32)
            self.assertEqual(pages[1]["image_stats"]["feature_dim"], 32)


class HybridPipelineTests(unittest.TestCase):
    def _make_pdf(self, folder: Path) -> Path:
        pdf_path = folder / "policy.pdf"
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text(
            (72, 72),
            "Policy Declarations\nCollision Deductible: $500\nComprehensive Deductible: $250\nPremium: $1200",
        )
        page2 = doc.new_page()
        page2.insert_text(
            (72, 72),
            "Section II - Liability Coverages\nCoverage E - Personal Liability\nWe do not cover electronic data loss.",
        )
        page3 = doc.new_page()
        page3.insert_text(
            (72, 72),
            "Cyber Liability Endorsement HO-123\nThis endorsement modifies the policy to provide cyber liability coverage subject to the limit shown in the declarations.",
        )
        page4 = doc.new_page()
        page4.insert_text(
            (72, 72),
            "Definitions\nElectronic data means information stored on computers or mobile devices.",
        )
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def _build_pipeline(self, root: Path, enable_image_signal: bool = True, retrieval_mode: str = "hybrid_multimodal"):
        docs_dir = root / "docs"
        index_dir = root / "index"
        docs_dir.mkdir()
        self._make_pdf(docs_dir)
        config = ModelConfig(
            index_dir=index_dir,
            retrieval_mode=retrieval_mode,
            corpus_source="documents",
            enable_image_signal=enable_image_signal,
            pdf_render_dir=root / "rendered_pages",
        )
        pipeline = DocumentRetrievalPipeline(config)
        pipeline.build_index(docs_dir)
        return pipeline, docs_dir, index_dir

    def test_query_understanding_detects_insurance_needs(self):
        understanding = understand_query(
            "Does the cyber liability endorsement add back coverage and what limit is shown on the declarations page?"
        )
        self.assertEqual(understanding.intent, "coverage_check")
        self.assertTrue(understanding.needs_endorsement_check)
        self.assertTrue(understanding.needs_limit)
        self.assertTrue(understanding.needs_table_lookup)
        self.assertTrue(understanding.needs_declarations)
        self.assertIn("cyber_liability", understanding.target_coverages)
        self.assertIn("declarations", understanding.preferred_document_types)
        self.assertIn("endorsement", understanding.preferred_document_types)
        self.assertIn("limit", understanding.preferred_field_types)
        self.assertTrue(any("Declarations" in section for section in understanding.preferred_sections))

    def test_rank_pages_returns_backward_compatible_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline, docs_dir, _ = self._build_pipeline(root)
            ranked = pipeline.rank_pages("What deductible is listed on the declarations page?", docs_dir, top_k=2)

            self.assertTrue(ranked)
            first = ranked[0]
            for key in ["source", "score", "retrieval_score", "rerank_score", "page_number", "text_snippet"]:
                self.assertIn(key, first)
            self.assertIn("snippet_support", first)
            self.assertIn("image_score", first)
            self.assertIn("document_type", first)
            self.assertIn("primary_clause_type", first)
            self.assertIn("section_anchor", first)

    def test_disable_image_signal_skips_auxiliary_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline, docs_dir, index_dir = self._build_pipeline(root, enable_image_signal=False, retrieval_mode="hybrid_text")

            self.assertFalse((index_dir / "hybrid_page_image.npy").exists())
            self.assertEqual(
                pipeline.retrieve_image_candidates(
                    "What deductible is listed?",
                    docs_dir,
                    page_sources={"policy.pdf#page=1"},
                ),
                {},
            )

    def test_multimodal_reranker_can_rescue_page_order(self):
        config = ModelConfig(image_signal_weight=0.5, enable_image_signal=True)
        pipeline = DocumentRetrievalPipeline(config)
        question = "What deductible is listed on the declarations page?"
        candidates = [
            {
                "record_id": "a",
                "record_type": "page",
                "source": "doc.pdf#page=1",
                "text": "The deductible is $500.",
                "page_number": 1,
                "rrf_score": 0.5,
                "retrieval_score": 0.5,
                "dense_rank": 1,
                "sparse_rank": 1,
            },
            {
                "record_id": "b",
                "record_type": "page",
                "source": "doc.pdf#page=2",
                "text": "The deductible is $500.",
                "page_number": 2,
                "rrf_score": 0.5,
                "retrieval_score": 0.5,
                "dense_rank": 1,
                "sparse_rank": 1,
            },
        ]

        no_image = pipeline.rerank_multimodal_candidates(question, candidates, image_scores={})
        with_image = pipeline.rerank_multimodal_candidates(
            question,
            candidates,
            image_scores={"doc.pdf#page=2": 1.0},
        )

        self.assertEqual(no_image[0]["source"], "doc.pdf#page=1")
        self.assertEqual(with_image[0]["source"], "doc.pdf#page=2")

    def test_build_index_writes_table_and_graph_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _, _, index_dir = self._build_pipeline(root)
            self.assertTrue((index_dir / "hybrid_tables.jsonl").exists())
            self.assertTrue((index_dir / "hybrid_tables_sparse.json").exists())
            self.assertTrue((index_dir / "hybrid_graph.jsonl").exists())

    def test_table_records_extract_numeric_fields(self):
        table_records = build_table_records(
            [
                {
                    "doc_id": "policy",
                    "page_key": "policy::p0001",
                    "source": "policy.pdf#page=1",
                    "page_number": 1,
                    "document_type": "declarations",
                    "text": "Collision Deductible: $500\nPremium: $1200",
                }
            ]
        )
        field_types = {record["field_type"] for record in table_records}
        self.assertIn("deductible", field_types)
        self.assertIn("premium", field_types)
        deductible_record = next(record for record in table_records if record["field_type"] == "deductible")
        self.assertEqual(deductible_record["normalized_field_name"], "collision deductible")
        self.assertEqual(deductible_record["normalized_field_value"], "$500")
        self.assertEqual(deductible_record["numeric_value"], 500.0)
        self.assertEqual(deductible_record["value_unit"], "currency")

    def test_graph_expansion_surfaces_endorsement_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline, docs_dir, _ = self._build_pipeline(root)
            merged = pipeline.merge_candidates(
                "Does the cyber liability endorsement modify the policy coverage?",
                docs_dir,
                top_k=5,
            )
            sources = {candidate.get("source") for candidate in merged}
            self.assertIn("policy.pdf#page=3", sources)
            graph_candidate = next(candidate for candidate in merged if candidate.get("source") == "policy.pdf#page=3")
            if graph_candidate.get("record_type") == "graph_page":
                self.assertIn("graph_reason", graph_candidate)

    def test_query_structured_includes_query_understanding_and_caveats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline, docs_dir, _ = self._build_pipeline(root)
            result = pipeline.query_structured(
                "What deductible is listed on the declarations page?",
                docs_dir,
                top_k=3,
                force_extractive=True,
            )
            self.assertIn("query_understanding", result)
            self.assertIn("caveats", result)
            self.assertEqual(result.get("limit"), "$500")
            self.assertEqual(result.get("coverage"), "Collision")
            self.assertEqual(result.get("evidence_role"), "declarations:deductible")
            self.assertEqual(result.get("citations", [])[0].get("section_anchor"), "Policy Declarations")

    def test_query_structured_reports_override_conflict_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pipeline, docs_dir, _ = self._build_pipeline(root)
            result = pipeline.query_structured(
                "Does the cyber liability endorsement modify the exclusion for electronic data loss?",
                docs_dir,
                top_k=4,
                force_extractive=True,
            )
            self.assertEqual(result.get("query_understanding", {}).get("target_coverages"), ["cyber_liability"])
            self.assertTrue(result.get("conflict_notes"))
            self.assertTrue(result.get("conflicts"))
            self.assertEqual(result.get("override_summary", {}).get("type"), "endorsement_override")


if __name__ == "__main__":
    unittest.main()
