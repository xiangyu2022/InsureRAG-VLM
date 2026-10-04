"""Pure answer scoring and prompt construction; no training dependencies."""
import re
from collections import Counter
import numpy as np

SYSTEM = ("Answer the insurance question using ONLY the supplied evidence. "
          "Give a concise supported answer and end with 'Source: ' followed by the supplied source exactly. "
          "If the evidence does not support the question, output only INSUFFICIENT_EVIDENCE. "
          "Do not infer coverage from general knowledge.")


def prompt_ids(tokenizer, row):
    if "input_source" in row:
        system = ("Answer the insurance question using ONLY the supplied evidence. "
                  "Give a concise supported answer and end with 'Source: ' followed by one SOURCE identifier "
                  "from the evidence exactly. If the evidence does not support the question, output only "
                  "INSUFFICIENT_EVIDENCE. Do not infer coverage from general knowledge.")
        user = f"Evidence:\n{row['evidence']}\nQuestion: {row['question']}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        f"Evidence:\n{row['evidence']}\nSource: {row.get('input_source', row['source'])}\nQuestion: {row['question']}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]



def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_f1(prediction, reference):
    a, b = Counter(words(prediction)), Counter(words(reference))
    overlap = sum((a & b).values())
    return 2 * overlap / (sum(a.values()) + sum(b.values())) if a or b else 1.0


def score(row, prediction):
    content = re.split(r"\bSource\s*:", prediction, flags=re.I)[0].strip()
    abstain = bool(re.search(r"INSUFFICIENT_EVIDENCE|insufficient evidence|cannot (?:confirm|determine)|does not (?:provide|support|mention)|not (?:specified|stated|supported)", prediction, re.I))
    citations = re.findall(r"\bSource\s*:\s*(.+)", prediction, re.I)
    correct_citation = bool(citations) and citations[-1].strip().rstrip(".") == row["source"].rstrip(".")
    tokens = words(content)
    evidence_tokens = set(words(row["evidence"]))
    coverage = sum(t in evidence_tokens for t in tokens) / max(1, len(tokens))
    numbers = set(re.findall(r"\d[\d,.]*(?:%|\b)", content))
    evidence_numbers = set(re.findall(r"\d[\d,.]*(?:%|\b)", row["evidence"]))
    return {"content_f1": content_f1(content, row["reference_content"]) if row["answerable"] else None,
            "abstains": abstain, "has_citation": bool(citations), "citation_correct": correct_citation,
            "evidence_token_coverage": coverage if not abstain else None,
            "numbers_supported": numbers <= evidence_numbers if not abstain else None,
            "lexical_support_proxy": bool(correct_citation and coverage >= 0.8 and numbers <= evidence_numbers and not abstain)}


def aggregate(rows):
    answerable = [x for x in rows if x["answerable"]]
    tp = sum(x["abstains"] and not x["answerable"] for x in rows)
    fp = sum(x["abstains"] and x["answerable"] for x in rows)
    fn = sum(not x["abstains"] and not x["answerable"] for x in rows)
    cited = sum(x["has_citation"] for x in rows)
    return {"n": len(rows), "n_answerable": len(answerable), "n_unsupported": len(rows) - len(answerable),
            "answerable_content_f1": sum(x["content_f1"] for x in answerable) / max(1, len(answerable)),
            "citation_precision_exact_source": sum(x["citation_correct"] for x in rows) / max(1, cited),
            "answerable_citation_rate": sum(x["citation_correct"] for x in answerable) / max(1, len(answerable)),
            "answerable_lexical_support_proxy": sum(x["lexical_support_proxy"] for x in answerable) / max(1, len(answerable)),
            "abstention_precision": tp / max(1, tp + fp), "abstention_recall": tp / max(1, tp + fn),
            "abstention_counts": {"tp": tp, "fp": fp, "fn": fn},
            "latency_p50_seconds": float(np.percentile([x["seconds"] for x in rows], 50)),
            "latency_p95_seconds": float(np.percentile([x["seconds"] for x in rows], 95)),
            "generated_tokens": sum(x["generated_tokens"] for x in rows)}
