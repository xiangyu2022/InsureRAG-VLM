import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.insurerag_vlm.training_data import TrainingCorpusBuildConfig, build_training_corpora


class TrainingCorpusBuilderTests(unittest.TestCase):
    def _make_pdf(self, folder: Path) -> Path:
        pdf_path = folder / "policy.pdf"
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text(
            (72, 72),
            "Policy Declarations\nCollision Deductible: $500\nLiability Limit: $300,000",
        )
        page2 = doc.new_page()
        page2.insert_text(
            (72, 72),
            "Section II - Liability Coverages\nCoverage E - Personal Liability\nWe do not cover electronic data loss.",
        )
        doc.save(pdf_path)
        doc.close()
        return pdf_path

    @staticmethod
    def _write_jsonl(records, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_build_training_corpora_writes_expected_manifests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            self._make_pdf(docs_dir)

            qa_path = root / "qa_pairs.jsonl"
            hard_negatives_path = root / "hard_negatives.jsonl"
            sft_path = root / "sft_dataset.jsonl"

            self._write_jsonl(
                [
                    {
                        "qa_id": "qa_train_1",
                        "question": "What is the collision deductible?",
                        "answer": "$500",
                        "evidence_sources": ["policy.pdf#page=1"],
                        "evidence_text": "Collision Deductible: $500",
                        "answerable": True,
                        "split": "train",
                    },
                    {
                        "qa_id": "qa_valid_1",
                        "question": "What exclusions are listed in the policy?",
                        "answer": "Electronic data loss is excluded.",
                        "evidence_sources": ["policy.pdf#page=2"],
                        "evidence_text": "We do not cover electronic data loss.",
                        "answerable": True,
                        "split": "train",
                    },
                    {
                        "qa_id": "qa_test_unsupported",
                        "question": "Does this policy include cyber liability coverage?",
                        "answer": "",
                        "evidence_sources": [],
                        "evidence_text": "",
                        "answerable": False,
                        "split": "test",
                    },
                ],
                qa_path,
            )
            self._write_jsonl(
                [
                    {
                        "qa_id": "qa_train_1",
                        "negative_page_id": "policy_p0002",
                        "negative_source": "policy.pdf#page=2",
                        "rank": 1,
                        "negative_type": "lexical_hard_negative",
                    }
                ],
                hard_negatives_path,
            )
            self._write_jsonl(
                [
                    {
                        "record_id": "sft_train",
                        "instruction": "Answer from evidence.",
                        "question": "What is the collision deductible?",
                        "evidence": "Collision Deductible: $500",
                        "answer": "The deductible is $500. Source: policy.pdf, Page 1",
                        "source": "policy.pdf#page=1",
                        "answerable": True,
                    },
                    {
                        "record_id": "sft_valid",
                        "instruction": "Answer from evidence.",
                        "question": "What exclusions are listed in the policy?",
                        "evidence": "We do not cover electronic data loss.",
                        "answer": "Electronic data loss is excluded. Source: policy.pdf, Page 2",
                        "source": "policy.pdf#page=2",
                        "answerable": True,
                        "split": "train",
                    },
                ],
                sft_path,
            )

            result = build_training_corpora(
                TrainingCorpusBuildConfig(
                    data_folder=docs_dir,
                    output_dir=root / "training",
                    qa_path=qa_path,
                    hard_negatives_path=hard_negatives_path,
                    sft_dataset_path=sft_path,
                    index_dir=root / "index",
                    retrieval_model="local-hashing",
                    enable_image_signal=False,
                )
            )

            self.assertTrue(Path(result["retrieval_train_path"]).exists())
            self.assertTrue(Path(result["rag_sft_train_path"]).exists())
            self.assertTrue(Path(result["calibration_dev_path"]).exists())

            retrieval_train = [json.loads(line) for line in Path(result["retrieval_train_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            rag_sft_train = [json.loads(line) for line in Path(result["rag_sft_train_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(retrieval_train[0]["question"], "What is the collision deductible?")
            self.assertTrue(retrieval_train[0]["hard_negative_texts"])
            self.assertEqual(rag_sft_train[0]["dataset_variant"], "retrieval_conditioned")
            self.assertIn("retrieval_context_sources", rag_sft_train[0])

    def test_build_training_corpora_normalizes_url_page_one_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "curated"
            data_dir.mkdir()

            pages_path = data_dir / "rag_pages.jsonl"
            snippets_path = data_dir / "rag_snippets.jsonl"
            corpus_path = data_dir / "rag_corpus.jsonl"
            sft_path = data_dir / "sft_dataset.jsonl"
            qa_path = root / "qa_pairs.jsonl"
            hard_negatives_path = root / "hard_negatives.jsonl"

            self._write_jsonl(
                [
                    {
                        "record_id": "page1",
                        "doc_id": "md_doi_auto_declarations_page",
                        "page": 1,
                        "citation": "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx",
                        "text": "Replacement cost is the amount needed to replace property.",
                    }
                ],
                pages_path,
            )
            self._write_jsonl(
                [
                    {
                        "record_id": "snippet1",
                        "doc_id": "md_doi_auto_declarations_page",
                        "page": 1,
                        "parent_page_id": "md_doi_auto_declarations_page::p0001",
                        "citation": "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx",
                        "text": "Replacement cost is the amount needed to replace property.",
                    }
                ],
                snippets_path,
            )
            self._write_jsonl([], corpus_path)
            self._write_jsonl(
                [
                    {
                        "qa_id": "qa_url_1",
                        "question": "What does replacement cost mean in this insurance context?",
                        "answer": "Replacement cost is the amount needed to replace property.",
                        "evidence_sources": [
                            "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx#page=1"
                        ],
                        "evidence_text": "Replacement cost is the amount needed to replace property.",
                        "answerable": True,
                        "split": "train",
                    }
                ],
                qa_path,
            )
            self._write_jsonl([], hard_negatives_path)
            self._write_jsonl(
                [
                    {
                        "record_id": "sft_url_1",
                        "instruction": "Answer from evidence.",
                        "question": "What does replacement cost mean in this insurance context?",
                        "evidence": "Replacement cost is the amount needed to replace property.",
                        "answer": "Replacement cost is the amount needed to replace property.",
                        "source": "https://insurance.maryland.gov/Consumer/Pages/Auto-Understanding-Declarations.aspx#page=1",
                        "answerable": True,
                    }
                ],
                sft_path,
            )

            result = build_training_corpora(
                TrainingCorpusBuildConfig(
                    data_folder=data_dir,
                    output_dir=root / "training",
                    qa_path=qa_path,
                    hard_negatives_path=hard_negatives_path,
                    sft_dataset_path=sft_path,
                    index_dir=root / "index",
                    retrieval_model="local-hashing",
                    corpus_source="curated",
                    enable_image_signal=False,
                )
            )

            retrieval_train = [
                json.loads(line)
                for line in Path(result["retrieval_train_path"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                retrieval_train[0]["gold_page_keys"],
                ["md_doi_auto_declarations_page::p0001"],
            )
            self.assertTrue(retrieval_train[0]["positive_page_texts"])


if __name__ == "__main__":
    unittest.main()
