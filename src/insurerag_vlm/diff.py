import difflib
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ClauseChange:
    change_type: str
    text: str


@dataclass
class SentenceChange:
    change_type: str
    old_sentence: str
    new_sentence: str
    score: float


def split_sentences(text: str) -> List[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text.strip()) if sentence.strip()]
    return sentences if sentences else [text.strip()]


def compare_clause_diff(old_text: str, new_text: str) -> List[ClauseChange]:
    old_lines = [line.strip() for line in old_text.splitlines() if line.strip()]
    new_lines = [line.strip() for line in new_text.splitlines() if line.strip()]
    diff = difflib.ndiff(old_lines, new_lines)
    changes: List[ClauseChange] = []
    for token in diff:
        code = token[:2]
        text = token[2:].strip()
        if code == "- ":
            changes.append(ClauseChange(change_type="removed", text=text))
        elif code == "+ ":
            changes.append(ClauseChange(change_type="added", text=text))
        elif code == "? ":
            continue
        else:
            changes.append(ClauseChange(change_type="unchanged", text=text))
    return changes


def score_sentence_changes(old_text: str, new_text: str) -> List[SentenceChange]:
    old_sentences = split_sentences(old_text)
    new_sentences = split_sentences(new_text)
    matcher = difflib.SequenceMatcher(None, old_sentences, new_sentences)
    changes: List[SentenceChange] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for sentence in old_sentences[i1:i2]:
                changes.append(SentenceChange("unchanged", sentence, sentence, 1.0))
        elif tag == "replace":
            old_segment = old_sentences[i1:i2]
            new_segment = new_sentences[j1:j2]
            for old_sentence, new_sentence in zip(old_segment, new_segment):
                score = difflib.SequenceMatcher(None, old_sentence, new_sentence).ratio()
                changes.append(SentenceChange("modified", old_sentence, new_sentence, score))
            for old_sentence in old_segment[len(new_segment):]:
                changes.append(SentenceChange("removed", old_sentence, "", 0.0))
            for new_sentence in new_segment[len(old_segment):]:
                changes.append(SentenceChange("added", "", new_sentence, 0.0))
        elif tag == "delete":
            for sentence in old_sentences[i1:i2]:
                changes.append(SentenceChange("removed", sentence, "", 0.0))
        elif tag == "insert":
            for sentence in new_sentences[j1:j2]:
                changes.append(SentenceChange("added", "", sentence, 0.0))

    return changes


def summarize_clause_diff(old_text: str, new_text: str, top_n: int = 5) -> str:
    clause_changes = compare_clause_diff(old_text, new_text)
    sentence_changes = score_sentence_changes(old_text, new_text)
    counts = Counter(change.change_type for change in clause_changes)
    output: List[str] = [
        f"Added clauses: {counts['added']}",
        f"Removed clauses: {counts['removed']}",
        f"Unchanged clauses: {counts['unchanged']}",
        f"Modified sentences: {sum(1 for change in sentence_changes if change.change_type == 'modified')}",
        "",
        "Top sentence-level modifications:",
    ]

    modified_sentences = [change for change in sentence_changes if change.change_type == "modified"]
    modified_sentences.sort(key=lambda change: change.score)
    for change in modified_sentences[:top_n]:
        output.append(
            f"- score={change.score:.3f} | old={change.old_sentence!r} | new={change.new_sentence!r}"
        )

    return "\n".join(output)


def render_clause_diff(changes: List[ClauseChange]) -> str:
    output_lines: List[str] = []
    for change in changes:
        prefix = {
            "added": "+",
            "removed": "-",
            "unchanged": " ",
        }.get(change.change_type, " ")
        output_lines.append(f"{prefix} {change.text}")
    return "\n".join(output_lines)
