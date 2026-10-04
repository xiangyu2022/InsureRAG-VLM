"""Produce an auditable paired report only after all required outputs exist."""
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("reports/local_20261004")


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_rows(name):
    return [json.loads(x) for x in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def main():
    before = read_json("base_test_metrics.json")
    after = read_json("adapter_test_metrics.json")
    training = read_json("training_summary.json")
    audit = read_json("data_audit.json")
    source = read_json("Qwen3.5-2B_source.json")
    baseline = {r["id"]: r for r in read_rows("base_test_predictions.jsonl")}
    adapted = {r["id"]: r for r in read_rows("adapter_test_predictions.jsonl")}
    test_rows = {r["id"]: r for r in read_rows("test.jsonl")}
    assert baseline.keys() == adapted.keys() == test_rows.keys()
    groups = defaultdict(list)
    changes = []
    for key, row in test_rows.items():
        a, b = baseline[key], adapted[key]
        if row["answerable"]:
            delta = b["content_f1"] - a["content_f1"]
            groups[row["doc_group"]].append(delta)
            changes.append({"id": key, "doc_id": row["doc_id"], "delta_content_f1": delta,
                            "question": row["question"], "reference": row["reference_content"],
                            "base": a["prediction"], "adapter": b["prediction"]})
    rng = random.Random(20261004)
    keys = sorted(groups)
    boot = []
    for _ in range(2000):
        sample = [value for group in rng.choices(keys, k=len(keys)) for value in groups[group]]
        boot.append(sum(sample) / len(sample))
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    comparison = {"base": before, "adapter": after,
                  "delta_answerable_content_f1": after["answerable_content_f1"] - before["answerable_content_f1"],
                  "document_cluster_bootstrap_95pct_interval": ci,
                  "bootstrap_groups": len(groups), "bootstrap_resamples": 2000,
                  "paired_answerable_improved": sum(x["delta_content_f1"] > 1e-8 for x in changes),
                  "paired_answerable_worse": sum(x["delta_content_f1"] < -1e-8 for x in changes),
                  "paired_answerable_tied": sum(abs(x["delta_content_f1"]) <= 1e-8 for x in changes),
                  "scope": "synthetic labels, provided evidence, text-only adaptation; not production accuracy",
                  "training": training}
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    changes.sort(key=lambda x: x["delta_content_f1"])
    (ROOT / "largest_answer_regressions.json").write_text(json.dumps(changes[:10], indent=2, ensure_ascii=False), encoding="utf-8")
    rows = [("Answerable content token F1", "answerable_content_f1"),
            ("Exact-source citation precision", "citation_precision_exact_source"),
            ("Answerable exact-citation rate", "answerable_citation_rate"),
            ("Answerable lexical-support proxy", "answerable_lexical_support_proxy"),
            ("Synthetic abstention precision", "abstention_precision"),
            ("Synthetic abstention recall", "abstention_recall"),
            ("Latency p50 seconds", "latency_p50_seconds"),
            ("Latency p95 seconds", "latency_p95_seconds")]
    lines = ["# Local Qwen3.5-2B BF16 LoRA experiment", "",
             f"Report generated: {datetime.now(timezone.utc).isoformat()}", "",
             "## Outcome", "",
             f"Completed {training['steps']} optimizer steps over {training['samples_seen']} sample presentations. "
             "Results below are paired on the same frozen 96-example test split; 62 answerable and 34 synthetic unsupported examples. "
             "Test evidence comes from 9 documents excluded from adaptation training. This is a given-evidence, synthetic-label experiment, not a production or human-reviewed benchmark.", "",
             "| Metric | Official base | Adapter |", "| --- | ---: | ---: |"]
    lines.extend(f"| {name} | {before[key]:.4f} | {after[key]:.4f} |" for name, key in rows)
    if (ROOT / "adapter_retrieved_context_test_metrics.json").exists():
        rag_base = read_json("base_retrieved_context_test_metrics.json")
        rag_adapter = read_json("adapter_retrieved_context_test_metrics.json")
        lines += ["", f"**Do not promote this adapter into the end-to-end answer path.** "
                  f"In the corrected paired retrieved-context diagnostic, content F1 changes from "
                  f"{rag_base['answerable_content_f1']:.4f} to {rag_adapter['answerable_content_f1']:.4f}. "
                  "The primary improvement above applies to supplied-evidence adaptation only; the existing retrieval/packing path remains inadequate."]
    lines += ["", f"Answerable F1 delta: {comparison['delta_answerable_content_f1']:+.4f}; "
              f"document-cluster bootstrap 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}] "
              f"over {len(groups)} document groups. The small, synthetic sample limits generalization.",
              f"Paired answerable cases: {comparison['paired_answerable_improved']} improved, "
              f"{comparison['paired_answerable_worse']} worse, {comparison['paired_answerable_tied']} tied.", "",
              "## Training evidence", "",
              f"- Model: `{source['model_id']}` at `{source['revision']}` (Apache-2.0).",
              "- GPU: RTX 4070 Laptop 8GB; BF16 LoRA; text-only inputs; frozen visual encoder.",
              "- 600 training rows; rank 8, alpha 16; batch 1, accumulation 4; two epochs; seed 20261004.",
              f"- Trainable parameters: {training['trainable_parameters']:,}; peak allocated GPU memory: {training['peak_allocated_mb']:.1f} MiB.",
              f"- Training wall time: {training['seconds']:.1f}s; mean training loss: {training['mean_train_loss']:.4f}.",
              f"- Dev token NLL: base {training['initial_dev_nll']:.4f}; selected {training['best_dev_nll']:.4f}.",
              f"- Selected adapter: `{training['selected_adapter']}`; final candidate: `{training['final_adapter']}`.",
              "- Selection uses dev NLL only and includes the base candidate. If selected_adapter is null, the adapter did not beat base under the frozen selection rule.", "",
              "## Data, evaluation and limitations", "",
              f"- Curated input: {audit['input_rows']} rows across {audit['original_documents']} source identifiers. "
              f"Exact/near-duplicate evidence groups reduced this to {audit['connected_document_groups']} independent splitting groups.",
              f"- Filtering: `{json.dumps(audit['rejected'])}`. Split hashes and document assignments are in `data_audit.json`.",
              "- Questions, references and unsupported labels are rule-generated from public insurance sources. No human labeling or semantic correctness certification is claimed.",
              "- Content F1 excludes templated introductory text and citation suffixes. It still rewards matching synthetic extractive references; valid paraphrases can score lower.",
              "- Exact citation matching and lexical/number support are automatic proxies, not entailment judgments. Labels can be incomplete or semantically noisy.",
              "- No overlap of original document groups or exact evidence hashes is allowed between adaptation train/dev/test. Near duplicates below the lexical threshold may remain.",
              "- Inference is greedy with thinking disabled and a 128-token output cap for both models. Desktop load is uncontrolled, so latency is indicative.",
              "- This does not rerun historical trained-BGE scores or prove end-to-end retrieval gains.", "",
              "## Supporting artifacts", "",
              "- `frozen_experiment_plan.json`, `hardware.json`, `requirements-lock.txt`",
              "- `data_audit.json`, `train.jsonl`, `dev.jsonl`, `test.jsonl`",
              "- `training_log.jsonl`, `training_summary.json`, adapter checkpoints and optimizer state",
              "- `base_test_predictions.jsonl`, `adapter_test_predictions.jsonl`, `comparison.json`",
              "- `largest_answer_regressions.json`, `retrieval_baseline.json`, `unit_tests.log`",
              "- Reproduction instructions: `docs/local_experiment_20261004.md`", ""]
    for name in ("base_retrieved_context_test", "adapter_retrieved_context_test"):
        if (ROOT / f"{name}_metrics.json").exists():
            metrics = read_json(f"{name}_metrics.json")
            metrics["abstention_precision"] = None
            metrics["abstention_recall"] = None
            lines += [f"## Supplemental {name}", "", f"```json\n{json.dumps(metrics, indent=2)}\n```", "",
                      "This uses fixed retrieved/packed contexts and original synthetic positive labels only. Unsupported labels were excluded because their original labels only apply to supplied evidence, not the whole corpus. Abstention precision/recall are NOT MEASURED here: retrieval can remove support, and we have no new context-level answerability judgments. Raw metric files retain the aggregator's zero-denominator convention; null above is the appropriate interpretation. Gold page IDs are not supplied in the prompt. It is a retrieval-context diagnostic, not an additional independent benchmark.", ""]
    lines += ["## Supplemental prompt correction", "",
              "The initial supplemental prompt put a citation-selection instruction in a Source field while requiring that field to be copied exactly. The adapter often copied the instruction as its citation. Complete first-run artifacts are preserved as `prompt_v1_*` (base/adapter content F1 0.1373/0.0856; citation precision 0.3103/0). The corrected multi-source prompt omits that contradictory field; both base and fixed step-300 are rerun on the same contexts/decoding/scoring. Primary results are unaffected. See `supplemental_prompt_revision.json` and the protocol document for disclosure; no model tuning followed test inspection.", "",
              "## Failure analysis and decision", "",
              "Use this adapter as a local research candidate for concise source-formatted answers; do not replace a production answerer or claim general insurance accuracy from these labels.", "",
              "- 23/62 answerable primary-test cases regress in content F1. Example `a0c5012fc2fc17b97854`: asked to explain actual cash value, the adapter extracts a deductible definition instead. It passes exact citation and lexical support, demonstrating those proxies do not measure relevance.",
              "- Example `4a505aeba950d93a5046`: reference concerns HO-6 contents/interior coverage; the adapter extracts a generic suggestion to check covered/excluded perils. F1 decreases by 0.5771 despite a valid source string.",
              "- All questions/references and unsupported labels are synthetic. Perfect refusal precision/recall on 34 negatives does not establish robustness to genuine missing-evidence questions.",
              "- Independent audits find no forbidden document-group/hash/near-duplicate overlap under the frozen criteria, but train/test share 33 question templates. Semantic publication-version independence and pretraining contamination are unverified.",
              "- Base hits the 128-token cap on 5/96 primary-test outputs; adapter on 0/96. This is the same frozen generation protocol, but truncation and output brevity affect F1/citation comparisons.",
              "- Human relevance, semantic entailment, visual-input adaptation, production serving, robust latency and historical trained-BGE/QLoRA reruns are unmeasured.", "",
              "- Fixed retrieval contexts preserve the gold source for only 12/62 examples, although 34/62 have that source in top-5 retrieval. Role ordering and the character budget can exclude the relevant retrieved page. This diagnostic measures the existing packer as well as answer adaptation; it cannot isolate model quality.",
              "- CLI execution succeeds, but the travel-insurance smoke retrieves a travel page and packs auto-insurance text; an abstention is a functional smoke outcome, not proof of a useful travel answer.", "",
              "The fixed step-300 checkpoint was not changed after observing test results. Scoring/prompt code was extracted into a CPU-only module during primary inference without changing behavior; independent re-scoring matches saved outputs/metrics. Initial baseline/training invocation hashes were not captured; subsequent invocation hashes and final source/artifact hashes are provided without retroactively claiming that provenance.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
