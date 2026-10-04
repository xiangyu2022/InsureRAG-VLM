"""Audit and report the exploratory packing regression without semantic claims."""
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.insurerag_vlm.local_answer_metrics import score, aggregate

ROOT = Path("reports/packing_20261004")


def obj(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = obj("frozen_protocol.json")
    results, failures = {}, []
    for split in ("dev", "test"):
        results[split] = {}
        for packing in ("legacy", "balanced"):
            contexts = {r["id"]: r for r in rows(ROOT / f"{split}_{packing}.jsonl")}
            for model in ("base", "adapter"):
                name = f"{model}_{split}_{packing}"
                predictions = rows(ROOT / f"{name}_predictions.jsonl")
                assert len(predictions) == len(contexts) and {p["id"] for p in predictions} == contexts.keys()
                cited, mapped = 0, 0
                for prediction in predictions:
                    context = contexts[prediction["id"]]
                    for key, value in score(context, prediction["prediction"]).items():
                        assert prediction[key] == value, (name, prediction["id"], key)
                    citations = re.findall(r"\bSource\s*:\s*(.+)", prediction["prediction"], re.I)
                    cited += bool(citations)
                    mapped += bool(citations and citations[-1].strip().rstrip(".") in
                                   {s.rstrip(".") for s in context["packed_sources"]})
                    if split == "test" and packing == "balanced" and prediction["content_f1"] < .25:
                        failures.append({"id": prediction["id"], "model": model, "question": context["question"],
                            "reference": context["reference_content"], "prediction": prediction["prediction"],
                            "gold_in_top5": context["gold_in_top5"], "gold_source_in_context": context["gold_source_in_context"],
                            "reference_span_in_context": context["reference_span_in_context"],
                            "note": "Low lexical F1 is not an independent semantic error label."})
                metrics = obj(f"{name}_metrics.json")
                assert aggregate(predictions) == metrics
                metrics = {**metrics, "abstention_precision": None, "abstention_recall": None,
                           "abstaining_outputs": sum(p["abstains"] for p in predictions),
                           "citation_resolves_to_packed_source": mapped / max(1, cited),
                           "citation_resolution_counts": {"resolved": mapped, "cited_outputs": cited}}
                metrics.pop("abstention_counts")
                if split == "test" and packing == "legacy":
                    original = {r["id"]: r for r in rows(Path("reports/local_20261004") / f"{model}_retrieved_context_test_predictions.jsonl")}
                    metrics["identical_predictions_to_first_round_control"] = sum(
                        p["prediction"] == original[p["id"]]["prediction"] for p in predictions)
                results[split][f"{model}_{packing}"] = metrics
    comparison = {"scope": plan["scope"], "metrics": results,
                  "packing": {s: obj(f"{s}_packing_summary.json") for s in ("dev", "test")},
                  "prediction_rescore": "all saved scores match final pure scorer",
                  "new_confirmatory_holdout": False, "training_performed": False}
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (ROOT / "failure_examples.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Packing repair: exploratory regression", "",
             "**These test examples were previously observed. This is an engineering regression, not a new independent benchmark. No training or checkpoint selection occurred.**", "",
             "The legacy packer reordered pages by role, copied long raw snippets, then trimmed the whole concatenation. The new packer preserves retrieval rank, deduplicates query-selected sentence fragments, allocates a separate budget per source, and records exact source/fragment associations. The HF path checks the complete Qwen-tokenized prompt against 2,048 input tokens, reserves 128 output tokens, and verifies the model context limit without tokenizer truncation.", "",
             "## Evidence availability", "",
             "| Split | Packing | Top-5 gold source | Packed gold source | Exact reference span | Max prompt tokens |", "|---|---|---:|---:|---:|---:|"]
    for split in ("dev", "test"):
        for packing in ("legacy", "balanced"):
            m = comparison["packing"][split][packing]
            lines.append(f"| {split} ({m['rows']}) | {packing} | {m['gold_in_top5']} | {m['gold_source_in_context']} | {m['reference_span_in_context']} | {m['max_prompt_tokens']} |")
    lines += ["", "Source presence and exact-span retention are proxies; neither proves relevance or completeness. Gold is used for these measurements only after both contexts are produced, never for page/fragment selection.", "",
              "## Paired generation", "", "| Split | Model / packing | Content F1 | Gold-source precision | Packed-source resolution | Lexical-support proxy | Abstentions |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for split, metrics in results.items():
        for name, m in metrics.items():
            lines.append(f"| {split} | {name} | {m['answerable_content_f1']:.4f} | {m['citation_precision_exact_source']:.4f} | {m['citation_resolves_to_packed_source']:.4f} | {m['answerable_lexical_support_proxy']:.4f} | {m['abstaining_outputs']} |")
    lines += ["", "All original positive examples are retained (25 dev, 62 exposed-test), with the same document-scoped questions. No topic-based filtering or favorable-subset headline is used. These queries include a document identifier as in the initial benchmark, not a gold page or gold answer. The unchanged base and fixed step-300 use greedy decoding with thinking disabled and the same output limit.", "",
              "## Interpretation and limits", "",
              "- Separate retrieval misses (gold absent from top-5), packing losses (retrieved gold source omitted), and possible generation errors. Even when the source is present, an absent reference span is not proof that the packed evidence is unanswerable.",
              "- Context-level abstention precision/recall are NOT MEASURED. Original positive labels do not transfer automatically to truncated/retrieved contexts. Raw aggregator zero-denominator values are not valid refusal-quality scores here.",
              "- Some synthetic questions ask only to summarize evidence but their references name a specific definition from a long glossary. Reference F1 cannot certify unrestricted RAG correctness for such under-specified queries.",
              "- Citation resolution only checks that a generated identifier names an actually packed source; gold-source matching and lexical coverage do not establish entailment or factual correctness.",
              "- The final base-model CLI now retains/cites the travel-insurance page, but turns its rhetorical cruise/travel-agency example into a supposed prerequisite. This qualitative failure (`cli_review.json`) remains despite a valid citation. The adapter is not promoted; the CLI requires an explicit adapter argument.",
              "- The token capacity guard applies to the explicit local HF experiment/CLI. Other provider clients still need their own tokenizer-aware envelope; generic packing retains its character-budget fallback.",
              "- Sentence fragments may be shortened within a source; truncation is recorded in the packing audit. Preserved source IDs do not guarantee all qualifications survive a finite context budget.",
              "- New-holdout feasibility: 8 unseen-source candidates after excluding prior SFT and the inspected travel CLI document; 5 have high shared-clause/duplicate risk, leaving only 3 lexically unflagged sources with no independent labels. No new confirmatory holdout was manufactured.",
              "- Complete regression suite: 53 passed. Actual Qwen-tokenizer Unicode fixture retains three source identifiers within 400 prompt tokens. Real generated-PDF tests cover relevant-page retention and exact source mapping.", "",
              "## Reproduce and inspect", "",
              "Use `docs/packing_experiment_20261004.md` in a fresh checkout. The frozen protocol records source/data/checkpoint hashes. Existing first-round evidence under `reports/local_20261004` is untouched. `first_round_code.patch` preserves the pre-repair implementation.",
              "Supporting files: `frozen_protocol.json`, `holdout_feasibility.json`, `*_packing_audit.json`, per-example predictions, `failure_examples.json`, `comparison.json`, `unit_tests.log`, and `artifact_hashes.json`."]
    (ROOT / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    paths = set(ROOT.glob("*.*"))
    paths.discard(ROOT / "artifact_hashes.json")
    paths.update(Path(p) for p in plan["source_hashes"])
    paths.update([Path(__file__), Path("scripts/freeze_packing_protocol.py"), Path("scripts/audit_packing_holdout.py"),
                  Path("scripts/query_local_adapter.py"), Path("tests/test_context_packing.py"), Path("docs/packing_experiment_20261004.md")])
    (ROOT / "artifact_hashes.json").write_text(json.dumps({str(p): sha(p) for p in sorted(paths) if p.is_file()}, indent=2))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
