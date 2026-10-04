"""Exercise actual PDF -> indexes -> evaluator paths, without mocked rankings."""
import json
import tempfile
import unittest
from pathlib import Path

import fitz
import numpy as np

from src.insurerag_vlm.benchmark import _text_retrieval_metrics
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.data import load_documents
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline
from src.insurerag_vlm.preprocess import PageImagePreprocessConfig, preprocess_page_images
from src.insurerag_vlm.qa import compute_retrieval_metrics
from src.insurerag_vlm.retrieval_metrics import score_ranking
from src.insurerag_vlm.visual import build_visual_index, compute_visual_retrieval_metrics, visual_search, _page_text_embeddings


class RealRetrievalContractTests(unittest.TestCase):
    def test_reranker_rescue_outside_old_top5_candidate_pool(self):
        # A deliberately controlled index: the relevant rerank candidate is at
        # cosine rank 21, beyond the old top_k=5 pool, but inside top_k=10's pool.
        # No retrieval/reranking functions are mocked.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = "What is the collision deductible?"
            vector = _page_text_embeddings([query])[0]
            orthogonal = np.zeros_like(vector)
            orthogonal[np.where(vector == 0)[0][0]] = 1.0
            vectors = np.stack([vector] * 20 + [0.8 * vector + 0.6 * orthogonal])
            np.save(root / "visual_stub.npy", vectors)
            pages = [{"source": f"other{i}.pdf#page=1", "page_number": 1,
                      "page_id": f"other{i}", "text_layer": "General introduction."} for i in range(20)]
            pages.append({"source": "gold.pdf#page=1", "page_id": "gold", "page_number": 1,
                          "text_layer": "Declarations. Collision Deductible: $500."})
            (root / "visual_stub_pages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pages))
            five = visual_search(query, root, "visual_stub", 5)
            ten = visual_search(query, root, "visual_stub", 10)
            self.assertEqual(five[0]["source"], "gold.pdf#page=1")
            self.assertEqual(five, ten[:5])

    def test_pdf_identity_and_real_evaluation_entry_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "documents"
            # Same filename and identical bytes in different packets must remain distinct.
            for packet in ("alpha", "beta"):
                folder = corpus / packet
                folder.mkdir(parents=True)
                with fitz.open() as pdf:
                    for text in ("Liability coverage limit is $100000.", "Collision deductible is $500."):
                        pdf.new_page().insert_text((40, 40), text)
                    pdf.save(folder / "policy.pdf")
            docs = load_documents(corpus)
            rendered = load_documents(corpus, render_pdf_pages=True, pdf_render_dir=root / "text_renders")
            self.assertEqual(len({str(d.image_path) for d in rendered}), 4)
            self.assertTrue(all(d.image_path.exists() for d in rendered))
            result = preprocess_page_images(PageImagePreprocessConfig(corpus, root / "prep", render_dpi=40))
            pages = [json.loads(x) for x in result.page_manifest_path.read_text().splitlines()]
            self.assertEqual({d.metadata["source"] for d in docs}, {p["source"] for p in pages})
            self.assertEqual(len({p["page_id"] for p in pages}), 4)
            gold = {"question": "What is the liability coverage limit?", "answerable": True,
                    "evidence_sources": ["alpha/policy.pdf#page=1"]}
            wrong = next(p for p in pages if p["source"] == "beta/policy.pdf#page=1")
            self.assertEqual(score_ranking(gold, [wrong])["hit_at_5"], 0)
            qa_path = root / "qa.jsonl"
            qa_path.write_text(json.dumps(gold) + "\n", encoding="utf-8")
            pipeline = DocumentRetrievalPipeline(ModelConfig(
                index_dir=root / "text", corpus_source="documents", enable_image_signal=False,
                retrieval_model="local-hashing", vlm_model="local-extractive", use_hf_api=False))
            pipeline.build_index(corpus)
            qa = compute_retrieval_metrics(pipeline, corpus, qa_path)
            bench = _text_retrieval_metrics(pipeline, corpus, qa_path, 10)
            for backend in ("visual_stub", "local_image"):
                build_visual_index(result.page_manifest_path, root / backend, backend)
                visual = compute_visual_retrieval_metrics(qa_path, root / backend, backend)
                self.assertEqual(visual["recall_at_5"], 1.0)
                self.assertEqual(visual["hit_at_5"], qa.hit_at_5)
            self.assertEqual(qa.recall_at_5, 1.0)
            self.assertEqual(qa.recall_at_5, bench["recall_at_5"])

    def test_real_visual_and_hybrid_output_depth_preserves_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            with fitz.open() as pdf:
                for i in range(48):
                    kind = "Liability coverage exclusions" if i % 3 else "Declarations deductible limit"
                    pdf.new_page().insert_text((40, 40), f"{kind}\nPremium {i * 91}, coverage limit {i * 1000}.")
                pdf.save(corpus / "policy.pdf")
            prep = preprocess_page_images(PageImagePreprocessConfig(corpus, root / "prep", render_dpi=30))
            pipeline = DocumentRetrievalPipeline(ModelConfig(
                index_dir=root / "text", corpus_source="documents", enable_image_signal=False,
                retrieval_model="local-hashing", vlm_model="local-extractive", use_hf_api=False,
                page_top_k=6, candidate_pool_size=12))
            pipeline.build_index(corpus)
            for backend in ("visual_stub", "local_image"):
                build_visual_index(prep.page_manifest_path, root / backend, backend)
                for query in ("What is the deductible?", "What liability exclusions apply?", "What is the premium?"):
                    a = visual_search(query, root / backend, backend, top_k=5)
                    b = visual_search(query, root / backend, backend, top_k=10)
                    self.assertEqual(a, b[:5])
                    self.assertEqual(pipeline.rank_pages(query, corpus, top_k=5),
                                     pipeline.rank_pages(query, corpus, top_k=10)[:5])


if __name__ == "__main__":
    unittest.main()
