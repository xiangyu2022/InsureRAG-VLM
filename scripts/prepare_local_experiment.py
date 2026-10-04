"""Freeze a leakage-aware, document-disjoint synthetic-label experiment.

All input text is from the checked-in public curated corpus. No human labels are
claimed. Exact and near-duplicate evidence connects documents BEFORE splitting.
"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261004
ROOT = Path("reports/local_20261004")


def norm(value):
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def doc_id(source):
    return re.sub(r"#page=\d+$", "", source.replace("\\", "/"))


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    source = Path("data/04_curated/sft_dataset.jsonl")
    rows = read(source)
    parents = {doc_id(r["source"]): doc_id(r["source"]) for r in rows}

    def find(d):
        while parents[d] != d:
            parents[d] = parents[parents[d]]
            d = parents[d]
        return d

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    evidence_docs = defaultdict(set)
    for r in rows:
        evidence_docs[norm(r["evidence"])].add(doc_id(r["source"]))
    evidence = sorted(evidence_docs)
    shingles = [set(zip(*(s.split()[i:] for i in range(5)))) for s in evidence]
    inverted = defaultdict(set)
    near_pairs = 0
    for i, text in enumerate(evidence):
        docs = sorted(evidence_docs[text])
        for d in docs[1:]:
            union(docs[0], d)
        candidates = set()
        for shingle in shingles[i]:
            candidates.update(inverted[shingle])
        for j in candidates:
            if len(shingles[i] & shingles[j]) / max(1, len(shingles[i] | shingles[j])) >= 0.85:
                near_pairs += 1
                union(docs[0], sorted(evidence_docs[evidence[j]])[0])
        for shingle in shingles[i]:
            inverted[shingle].add(i)
    groups = defaultdict(list)
    for d in sorted(parents):
        groups[find(d)].append(d)
    group_keys = sorted(groups)
    random.Random(SEED).shuffle(group_keys)
    n = len(group_keys)
    n_test = max(2, round(n * 0.2))
    n_dev = max(2, round(n * 0.15))
    group_split = {g: "test" if i < n_test else "dev" if i < n_test + n_dev else "train"
                   for i, g in enumerate(group_keys)}
    document_split = {d: group_split[find(d)] for d in parents}
    rejected = Counter()
    unique = {}
    for r in rows:
        evidence_text = " ".join(r["evidence"].split())
        if not 20 <= len(evidence_text.split()) <= 280:
            rejected["evidence_length_outside_20_280_words"] += 1
            continue
        if r["answerable"]:
            # Remove generated boilerplate before training AND scoring.
            body = r["answer"].rsplit(" Source:", 1)[0]
            quote = body.split(": ", 1)[-1].strip().rstrip(".")
            quote = quote.rstrip(".").strip()
            if not quote or norm(quote) not in norm(evidence_text):
                rejected["reference_not_verbatim_evidence"] += 1
                continue
            # Existing topic heuristics can generate mislabeled evidence.
            topic = str(r.get("topic", ""))
            terms = {"exclusion": ["exclu"], "deductible": ["deductib"], "premium": ["premium"],
                     "limit": ["limit"], "liability": ["liabil"], "endorsement": ["endorse"],
                     "actual cash value": ["actual cash value", "depreciat"],
                     "replacement cost": ["replacement cost"]}.get(topic)
            if terms and not any(t in evidence_text.lower() for t in terms):
                rejected["topic_missing_from_evidence"] += 1
                continue
            answer = quote + f". Source: {r['source']}"
        else:
            quote = "INSUFFICIENT_EVIDENCE"
            answer = "INSUFFICIENT_EVIDENCE"
        key = digest(norm(r["question"]) + "|" + norm(evidence_text))
        if key in unique:
            rejected["duplicate_question_evidence"] += 1
            continue
        unique[key] = {"id": key[:20], "original_record_id": r["record_id"],
                       "doc_id": doc_id(r["source"]), "doc_group": find(doc_id(r["source"])),
                       "split": document_split[doc_id(r["source"])], "source": r["source"],
                       "question": r["question"], "evidence": evidence_text,
                       "reference_content": quote, "target": answer, "answerable": r["answerable"],
                       "label_origin": "existing_rule_generated_filtered_not_human_reviewed",
                       "evidence_sha256": digest(norm(evidence_text)), "topic": r.get("topic")}
    counts_before_cap = {}
    selected = {}
    for split, cap in (("train", 600), ("dev", 48), ("test", 96)):
        pool = [r for r in unique.values() if r["split"] == split]
        counts_before_cap[split] = len(pool)
        # Round robin documents and label classes, deterministic and answer-blind.
        buckets = defaultdict(list)
        for r in pool:
            buckets[(r["doc_id"], r["answerable"])].append(r)
        for bucket in buckets.values():
            bucket.sort(key=lambda r: digest(str(SEED) + r["id"]))
        chosen = []
        while len(chosen) < cap and any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key] and len(chosen) < cap:
                    chosen.append(buckets[key].pop())
        selected[split] = chosen
        write(ROOT / f"{split}.jsonl", chosen)
    assert all(selected.values())
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        assert not ({r["doc_group"] for r in selected[a]} & {r["doc_group"] for r in selected[b]})
        assert not ({r["evidence_sha256"] for r in selected[a]} & {r["evidence_sha256"] for r in selected[b]})
    audit = {"seed": SEED, "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "input_rows": len(rows), "original_documents": len(parents), "connected_document_groups": n,
             "near_duplicate_evidence_pairs_at_0.85": near_pairs, "rejected": dict(rejected),
             "counts_before_cap": counts_before_cap, "document_split": document_split,
             "group_members": dict(groups), "splits": {s: {
                 "rows": len(rs), "answerable": sum(r["answerable"] for r in rs),
                 "unsupported": sum(not r["answerable"] for r in rs),
                 "documents": len({r["doc_id"] for r in rs}),
                 "sha256": hashlib.sha256((ROOT / f"{s}.jsonl").read_bytes()).hexdigest()}
                 for s, rs in selected.items()},
             "scope": "Public real-source evidence; synthetic question/reference and unsupported labels. Not a human benchmark.",
             "remaining_limitations": ["Semantic duplicates below lexical Jaccard threshold may remain.",
                                       "Topic and unsupported labels are heuristic, not independently verified.",
                                       "Questions can share templates across documents; supplied evidence disambiguates them."]}
    (ROOT / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("input_rows", "original_documents", "connected_document_groups", "rejected", "counts_before_cap", "splits")}, indent=2))


if __name__ == "__main__":
    main()
