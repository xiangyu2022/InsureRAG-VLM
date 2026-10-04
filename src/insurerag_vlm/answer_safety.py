"""Conservative, transparent checks for an explicit evidence-based decline.

This is a wording heuristic, not an entailment model. A detected decline must be
preserved through deterministic postprocessing instead of being repaired into
an apparent factual answer.
"""
import re


_ABSTENTION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\binsufficient(?:_|\s)+(?:evidence|information)\b",
    r"\b(?:cannot|can not|can't|unable to)\s+(?:determine|answer|confirm|establish|find|verify|support|infer|provide)\b",
    r"\b(?:cannot|can not|can't)\s+be\s+(?:determined|confirmed|established|inferred|verified|supported)\b",
    r"\b(?:not enough|not sufficient|no sufficient)\s+(?:evidence|information)\b",
    r"\b(?:provided|supplied|retrieved|available)\b.{0,100}\b(?:does not|doesn't|do not|don't)\s+(?:state|specify|contain|provide|mention|establish|show)\b",
    r"\b(?:the|this)\s+(?:document|policy|guide|evidence|excerpt|page|context)\b.{0,70}\b(?:does not|doesn't|do not|don't)\s+(?:state|specify|contain|provide|mention|establish|show)\b",
    r"\b(?:is|are|was|were)\s+not\s+(?:stated|specified|provided|shown|supported|available)\b",
    r"\b(?:not stated|not specified|not supported)\s+(?:in|by|from)\s+(?:the\s+)?(?:provided|supplied|retrieved|available|evidence|context|document|policy|guide)\b",
    r"\b(?:i|we)\s+(?:do not|don't)\s+know\b",
    # Declining an answer is different from a document not supporting an exclusion.
    r"\b(?:the|this)\s+(?:(?:provided|supplied|retrieved|available)\s+)?(?:evidence|context|text|document|guide)\s+(?:does not|doesn't)\s+support\s+(?:(?:a|an|the)\s+)?(?:(?:definitive|reliable|conclusive|supported|specific|single|exact)\s+)*(?:answer|conclusion|determination)\b",
    # Sentence-leading assertions avoid matching 'not impossible' or quoted advice
    # such as 'Do not assume it is impossible to determine the deductible'.
    r"(?:^|[.!?]\s+)(?:(?:therefore|thus|so),?\s+)?(?:(?:it|this|that)\s+is\s+)?impossible\s+to\s+(?:determine|answer|confirm|establish|verify|infer)\b",
    # Only a missing value/decision, not a normal exclusion or a known zero amount.
    r"^(?:\*\*|__)?no\s+[^.;:]{0,70}?\b(?:amount|limit|deductible|premium|date|deadline|period|benefit|payment|identifier|policy number|precedence|controlling version|coverage decision)\b[^.;:]{0,70}?\s+(?:is|are)\s+(?:stated|specified|provided|shown|established|recorded|available)\b",
    # A leading missing-information response is a decline. Do not match a
    # later caveat after a supported answer about a different requested field.
    r"^(?:based on\s+(?:the\s+)?(?:evidence|information|documents?|context)\s*(?:provided|supplied|available)?\s*,?\s*)?there\s+is\s+no\s+information\s+(?:about|on|regarding|to determine)\b",
))


def is_explicit_abstention(answer: str) -> bool:
    normalized = " ".join((answer or "").replace("’", "'").split())
    return any(pattern.search(normalized) for pattern in _ABSTENTION_PATTERNS)
