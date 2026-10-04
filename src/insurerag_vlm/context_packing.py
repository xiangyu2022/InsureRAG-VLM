"""Rank-preserving evidence packing with per-source budgets and prompt capacity checks.

No labels or gold source IDs are accepted. A token counter, when supplied, must
measure the COMPLETE formatted model prompt, including question and instructions.
"""
import re
from typing import Callable, Optional


def normalized(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clip(text, limit):
    if len(text) <= limit:
        return text
    prefix = text[:max(0, limit)]
    # Preserve an exact prefix; do not append invented completion text.
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip()


def pack_evidence(ranked_pages, answer_top_k, max_chars=3200, max_page_chars=900,
                  prompt_token_counter: Optional[Callable[[str], int]] = None,
                  max_prompt_tokens: Optional[int] = None, prefer_tables=False, question=""):
    if max_chars <= 0 or max_page_chars <= 0 or answer_top_k <= 0:
        raise ValueError("Packing capacities must be positive")
    if (prompt_token_counter is None) != (max_prompt_tokens is None):
        raise ValueError("Supply both complete-prompt token counter and token capacity")
    if prompt_token_counter and prompt_token_counter("") > max_prompt_tokens:
        raise ValueError("Question/instructions alone exceed the input token capacity")
    pages, seen, dropped = [], set(), []
    for page in ranked_pages:
        source = str(page.get("source", ""))
        if not source or "\n" in source or "\r" in source:
            raise ValueError("A page requires a nonempty single-line source identifier")
        if source in seen:
            continue
        seen.add(source)
        fragments = []
        text = normalized(page.get("text_snippet"))
        supports = [normalized(x) for x in page.get("snippet_support", []) if x]
        tables = []
        for field in page.get("table_fields", []) or []:
            name = normalized(field.get("normalized_field_name") or field.get("field_name"))
            value = normalized(field.get("normalized_field_value") or field.get("field_value"))
            if name and value:
                tables.append(("table", f"{name}: {value}"))
        # Document scoping identifies a packet, not answer content. Do not let
        # URL/state-name tokens rank navigation above the insurance question.
        content_question = re.sub(r"^In the document .+?,\s*", "", question, flags=re.I)
        stop = {"what", "which", "does", "that", "this", "with", "from", "about", "would", "should",
                "evidence", "insurance", "specific", "supported", "support", "point", "explained", "explain",
                "summarize", "information", "using", "page", "document", "policy", "coverage", "say", "how", "the"}
        terms = {w for w in re.findall(r"[a-z0-9]+", content_question.lower()) if len(w) > 2} - stop
        sentences = []
        for support in supports or [text]:
            for sentence in re.split(r"(?<=[.!?])\s+", support):
                sentence = sentence.strip()
                if sentence and sentence not in sentences:
                    sentences.append(sentence)
        if question:
            def relevance(item):
                index, sentence = item
                words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                score = 3 * len(terms & words)
                boilerplate = re.search(r"skip to main|privacy policy|site map|mega menu|all rights reserved|"
                                        r"contact.*(?:administration|department)|suite \d|800-\d|410-\d", sentence, re.I)
                score -= 6 * bool(boilerplate)
                score += .1 * len(words & {"premium", "deductible", "exclusion", "liability", "claim", "claims", "coverage", "insured"})
                return (-score, index)
            sentences = [sentence for _, sentence in sorted(enumerate(sentences), key=relevance)]
        candidates = [("text", s) for s in sentences if s]
        candidates = tables + candidates if prefer_tables else candidates + tables
        for kind, value in candidates:
            if any(value in old[1] for old in fragments):
                continue
            # Keep the already query-selected fragment before longer raw support.
            if kind == "text" and any(old[1] in value for old in fragments if old[0] == "text"):
                continue
            fragments.append((kind, value))
        if not fragments:
            dropped.append({"source": source, "reason": "no_evidence_text"})
            continue
        pages.append({"source": source, "fragments": fragments,
                      "header": f"SOURCE: {source}\n"})
        if len(pages) == answer_top_k:
            break

    def render(budgets):
        blocks, segments = [], []
        for page, budget in zip(pages, budgets):
            body, remaining = [], budget
            for kind, text in page["fragments"]:
                piece = clip(text, remaining - 3)
                if not piece:
                    continue
                if body and len(piece) < min(24, len(text)):
                    continue
                body.append("- " + piece)
                segments.append({"source": page["source"], "kind": kind, "text": piece,
                                 "fragment_prefix_chars": len(piece), "fragment_chars": len(text),
                                 "truncated": len(piece) < len(text)})
                remaining -= len(piece) + 3
                if remaining <= 3:
                    break
            blocks.append(page["header"] + "\n".join(body))
        return "\n---\n".join(blocks), segments

    def fits(budgets):
        context, _ = render(budgets)
        return len(context) <= max_chars and (prompt_token_counter is None or
                                              prompt_token_counter(context) <= max_prompt_tokens)

    budgets = [min(48, max_page_chars)] * len(pages)
    while pages and not fits(budgets):
        dropped.append({"source": pages.pop()["source"], "reason": "minimum_evidence_exceeds_capacity"})
        budgets.pop()
    if not pages:
        return {"context": "", "sources": [], "segments": [], "dropped": dropped,
                "prompt_tokens": prompt_token_counter("") if prompt_token_counter else None}
    # Round-robin growth prevents a long first source from consuming every slot.
    while True:
        grew = False
        for index in range(len(pages)):
            if budgets[index] >= max_page_chars:
                continue
            candidate = list(budgets)
            candidate[index] = min(max_page_chars, budgets[index] + 96)
            if fits(candidate):
                budgets, grew = candidate, True
        if not grew:
            break
    context, segments = render(budgets)
    tokens = prompt_token_counter(context) if prompt_token_counter else None
    assert len(context) <= max_chars
    assert tokens is None or tokens <= max_prompt_tokens
    assert all(any(s["source"] == p["source"] for s in segments) for p in pages)
    return {"context": context, "sources": [p["source"] for p in pages], "segments": segments,
            "dropped": dropped, "prompt_tokens": tokens, "page_budgets_chars": budgets}
