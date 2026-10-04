"""Hand-authored false-candidate diagnostic. No model or retrieval inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.insurerag_vlm.config import ModelConfig
from src.insurerag_vlm.hybrid_pipeline import DocumentRetrievalPipeline


CASES = [
    {
        "id": "G01", "category": "same_page_deductible_swap",
        "question": "What is the comprehensive deductible?",
        "evidence": "Collision deductible: $750. Comprehensive deductible: $1,500.",
        "false_candidate": "The comprehensive deductible is $750.",
        "why_false": "$750 belongs to collision; comprehensive is $1,500.",
    },
    {
        "id": "G02", "category": "amount_belongs_to_another_scheduled_item",
        "question": "What is the camera coverage limit?",
        "evidence": "Camera coverage limit: $2,000. Laptop coverage limit: $4,000.",
        "false_candidate": "The camera coverage limit is $4,000.",
        "why_false": "$4,000 belongs to the laptop; the camera limit is $2,000.",
    },
    {
        "id": "G03", "category": "per_day_vs_per_claim_unit",
        "question": "What is the travel reimbursement limit?",
        "evidence": "Travel reimbursement limit: $250 per day.",
        "false_candidate": "The travel reimbursement limit is $250 per claim.",
        "why_false": "The limit applies per day, not per claim.",
    },
    {
        "id": "G04", "category": "percent_vs_currency_unit",
        "question": "What is the hurricane deductible?",
        "evidence": "The hurricane deductible is 10% of the insured value.",
        "false_candidate": "The hurricane deductible is $10.",
        "why_false": "A percentage of insured value is not a fixed $10 deductible.",
    },
    {
        "id": "G05", "category": "minimum_maximum_value_swap",
        "question": "What is the maximum equipment coverage limit?",
        "evidence": "The minimum equipment coverage limit is $1,000. The maximum equipment coverage limit is $5,000.",
        "false_candidate": "The maximum equipment coverage limit is $1,000.",
        "why_false": "The evidence associates $1,000 with the minimum and $5,000 with the maximum.",
    },
    {
        "id": "G06", "category": "minimum_only_does_not_establish_maximum",
        "question": "What is the maximum equipment coverage limit?",
        "evidence": "The equipment coverage limit must be at least $1,000.",
        "false_candidate": "The maximum equipment coverage limit is $1,000.",
        "why_false": "A lower bound does not establish an upper bound.",
    },
    {
        "id": "G07", "category": "coverage_negation_reversal",
        "question": "Is water backup covered?",
        "evidence": "Water backup is not covered under this policy.",
        "false_candidate": "Water backup is covered under this policy.",
        "why_false": "The candidate reverses an explicit coverage negation.",
    },
    {
        "id": "G08", "category": "effective_expiration_date_swap",
        "question": "What effective date is stated for policy ZX-77?",
        "evidence": "Policy ZX-77. Effective date: June 2, 2027. Expiration date: June 2, 2028.",
        "false_candidate": "The effective date for policy ZX-77 is June 2, 2028.",
        "why_false": "The candidate substitutes the expiration date for the effective date.",
    },
    {
        "id": "G09", "category": "wrong_policy_identifier",
        "question": "What dwelling limit is declared for policy ZX-78?",
        "evidence": "Policy ZX-77. Dwelling limit: $280,000.",
        "false_candidate": "The dwelling limit for policy ZX-78 is $280,000.",
        "why_false": "The supplied page describes ZX-77, not the requested ZX-78.",
    },
    {
        "id": "G10", "category": "repair_estimate_as_unapproved_payment",
        "question": "What approved insurer payment is established for claim CL-Z under policy ZX-77?",
        "evidence": "Claim report CL-Z. Policy ZX-77. Repair estimate: $9,400. No coverage acceptance or approved insurer payment is recorded.",
        "false_candidate": "The approved insurer payment is $9,400.",
        "why_false": "A repair estimate is not an approved payment; the evidence explicitly says approval is absent.",
    },
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("cases.jsonl", "results.jsonl", "summary.json", "README.md")):
        raise RuntimeError("Diagnostic output already exists; use a fresh output directory.")
    source_paths = [ROOT / "src/insurerag_vlm" / name for name in ("hybrid_pipeline.py", "answer_safety.py", "config.py")]
    before = {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths}
    config = ModelConfig(vlm_model="local-extractive", retrieval_model="local-hashing",
                         abstain_threshold=0.20, citation_min_overlap=0.20, enable_image_signal=False)
    pipeline = DocumentRetrievalPipeline(config)
    results = []
    for case in CASES:
        source = f"synthetic-{case['id']}.pdf#page=1"
        context = f"SOURCE: {source}\nROLE: declarations / general\n{case['evidence']}"
        page = {"source": source, "score": 0.95, "document_type": "declarations",
                "primary_clause_type": "general", "text_snippet": case["evidence"], "table_fields": []}
        supplied = {"answer": case["false_candidate"] + "\n\nSOURCE: " + source,
                    "source_ranking": [page], "retrieval_context": context,
                    "generation_used": False, "answer_backend": "constructed-candidate-no-model",
                    "backend_metadata": {"generation_run": False}}
        helper_pass, helper_reason = pipeline._citation_support_details(
            case["question"], case["false_candidate"],
            [{"source": source, "evidence_text": case["evidence"], "document_type": "declarations"}],
            min_overlap=config.citation_min_overlap,
        )
        with patch.object(pipeline, "query_with_ranking", return_value=supplied), \
                patch.object(pipeline.vlm_client, "generate", side_effect=AssertionError("No model call permitted")):
            served = pipeline.query_structured(case["question"], Path("not-used"))
        results.append({**case, "candidate_is_deliberately_false": True, "source": source,
                        "helper_accepts": helper_pass, "helper_reason": helper_reason,
                        "served_accepts": not served["abstain"], "served_answer": served["answer"],
                        "served_reason": served["citation_support_reason"],
                        "answer_repaired": served["answer_repaired"], "conflicts": served["conflicts"],
                        "heuristic_evidence_score": served["confidence"],
                        "model_inference_performed": False, "retrieval_inference_performed": False})
    after = {path.relative_to(ROOT).as_posix(): sha(path) for path in source_paths}
    if before != after:
        raise RuntimeError("Runtime source changed during this diagnostic; rerun into a fresh directory.")
    summary = {
        "label": "Hand-authored synthetic false-candidate guard diagnostic; no model evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": len(results),
        "helper_false_acceptances": sum(row["helper_accepts"] for row in results),
        "served_false_acceptances": sum(row["served_accepts"] for row in results),
        "source_sha256": before, "script_sha256": sha(Path(__file__)),
        "model_inference_performed": False, "retrieval_inference_performed": False,
        "construction": "Ten deliberately false candidates and a manually supplied one-page ranking at score0.95 isolate guard behavior. No PDF ingestion or live retrieval is exercised.",
        "interpretation": "Hand-selected adversarial examples are not a population error-rate estimate. Acceptance here demonstrates a guard gap; rejection only checks this wording. No guard code or public/synthetic benchmark fixture was changed as part of this diagnostic.",
        "thresholds": {"citation_min_overlap": config.citation_min_overlap, "abstain_threshold": config.abstain_threshold},
    }
    for name, rows in (("cases.jsonl", CASES), ("results.jsonl", results)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Adversarial guard diagnostic v1", "",
             "**Synthetic, deliberately false candidates. No model inference or retrieval run.**", "",
             f"The citation helper accepted {summary['helper_false_acceptances']} of these10 false candidates; the served-answer path accepted {summary['served_false_acceptances']}. These hand-selected cases do not estimate a real-world error rate.", "",
             "The one-page ranking was supplied at a fixed heuristic score0.95 to isolate postprocessing. All evidence, candidate answers, falsity rationales, and results are retained in cases.jsonl/results.jsonl. No source or frozen fixture was changed.", "",
             "| Case | Deliberately false claim | Helper | Served path | Reason |", "| --- | --- | --- | --- | --- |"]
    for row in results:
        lines.append(f"| {row['id']} | {row['category'].replace('_', ' ')} | {'FALSE ACCEPT' if row['helper_accepts'] else 'reject'} | {'FALSE ACCEPT' if row['served_accepts'] else 'reject'} | {row['served_reason']} |")
    lines.extend(["", "The accepted cases show that matching terms, dates, and amounts is insufficient to establish field/value associations, units, bounds, or negation. High heuristic evidence scores can accompany false statements. These scores are not calibrated correctness probabilities.", "",
                  "The rejected controls show narrower checks for absent identifiers, unmatched currency/percentage forms, missing maximum evidence, and expressly unapproved payments. They do not establish general semantic reasoning.", "",
                  "See [known limitations](../../docs/KNOWN_LIMITATIONS.md) and [runtime follow-up](../packet_stress_v1/runtime_guard_followup.md). Reproduce with `python reports/guard_adversarial_v1/run.py --output <new-directory>`; existing reports are never overwritten.", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args().output)
