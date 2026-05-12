import unittest

from src.insurerag_vlm.validation import validate_curated_record_sets


def _minimal_records(unsupported_count=2):
    page = {
        "record_id": "rag_page::doc::p0001",
        "record_type": "page",
        "doc_id": "doc",
        "source_file": "doc.pdf",
        "page": 1,
        "text": "The deductible is $1,000.",
        "citation": "doc.pdf#page=1",
    }
    snippet = {
        "record_id": "rag_snippet::doc::p0001::c001",
        "record_type": "snippet",
        "parent_page_id": page["record_id"],
        "doc_id": "doc",
        "source_file": "doc.pdf",
        "page": 1,
        "text": "The deductible is $1,000.",
        "citation": "doc.pdf#page=1",
    }
    sft_records = [
        {
            "record_id": "sft::answerable",
            "instruction": "Answer from evidence.",
            "question": "What is the deductible?",
            "evidence": "The deductible is $1,000.",
            "answer": "The deductible is $1,000. Source: doc.pdf, Page 1",
            "source": "doc.pdf#page=1",
            "answerable": True,
        }
    ]
    for index in range(unsupported_count):
        sft_records.append(
            {
                "record_id": f"sft::unsupported::{index}",
                "instruction": "Answer from evidence.",
                "question": f"What is not stated {index}?",
                "evidence": "The deductible is $1,000.",
                "answer": "The evidence is insufficient to answer.",
                "source": "doc.pdf#page=1",
                "answerable": False,
            }
        )
    return {
        "rag_pages": [page],
        "rag_snippets": [snippet],
        "rag_corpus": [page, snippet],
        "sft_dataset": sft_records,
    }


class CuratedValidationTests(unittest.TestCase):
    def test_valid_curated_records_pass(self):
        result = validate_curated_record_sets(_minimal_records(), min_unsupported=2)

        self.assertTrue(result["passed"])
        self.assertEqual(result["counts"]["sft_dataset"], 3)
        self.assertEqual(result["quality"]["sft_answerable_count"], 1)

    def test_missing_answerable_evidence_fails(self):
        records = _minimal_records()
        records["sft_dataset"][0]["evidence"] = ""

        result = validate_curated_record_sets(records, min_unsupported=2)

        self.assertFalse(result["passed"])
        self.assertIn("empty_answerable_evidence", {error["type"] for error in result["errors"]})

    def test_too_few_unsupported_records_fails(self):
        result = validate_curated_record_sets(_minimal_records(unsupported_count=1), min_unsupported=2)

        self.assertFalse(result["passed"])
        self.assertIn("too_few_unsupported_records", {error["type"] for error in result["errors"]})


if __name__ == "__main__":
    unittest.main()
