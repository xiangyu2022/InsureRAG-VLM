import re
from typing import List, Set


_COVERAGE_PATTERNS = {
    "personal_liability": [r"personal liability", r"coverage e", r"liability coverage"],
    "medical_payments": [r"medical payments", r"coverage f"],
    "cyber_liability": [r"cyber", r"data breach", r"electronic data", r"privacy liability"],
    "property": [r"dwelling", r"coverage a", r"property damage", r"personal property"],
    "loss_of_use": [r"loss of use", r"additional living expense"],
    "comprehensive": [r"comprehensive"],
    "collision": [r"collision"],
    "uninsured_motorist": [r"uninsured motorist", r"underinsured motorist"],
    "water_backup": [r"water backup", r"sewer backup"],
    "earthquake": [r"earthquake"],
    "umbrella": [r"umbrella", r"excess liability"],
}

_FORM_CODE_RE = re.compile(r"\b[A-Z]{1,5}[- ]?\d{2,}[A-Z]?\b")
_GENERIC_SECTION_TOKENS = {
    "section", "coverage", "coverages", "policy", "form", "endorsement",
    "definitions", "definition", "conditions", "condition", "exclusions",
    "exclusion", "schedule", "liability", "general", "property", "part",
}


def coverage_tag_label(tag: str) -> str:
    return str(tag or "").replace("_", " ").strip().title()


def normalize_coverage_labels(tags: List[str]) -> List[str]:
    return [coverage_tag_label(tag) for tag in tags if str(tag or "").strip()]


def normalize_heading(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip(" :-")
    if not cleaned:
        return ""
    if cleaned.isupper() or cleaned.islower():
        parts = []
        for token in cleaned.split():
            if re.fullmatch(r"[IVXLCM]+", token):
                parts.append(token)
            elif re.fullmatch(r"[ivxlcm]+", token):
                parts.append(token.upper())
            elif token.isupper() and len(token) <= 3:
                parts.append(token)
            else:
                parts.append(token.title())
        cleaned = " ".join(parts)
    return cleaned


def extract_form_codes(text: str) -> List[str]:
    return sorted({match.group(0).replace(" ", "-") for match in _FORM_CODE_RE.finditer(text or "")})


def _looks_like_heading(line: str) -> bool:
    lowered = line.lower()
    if len(line) < 4 or len(line) > 120:
        return False
    if any(char.isdigit() for char in line) and "$" in line:
        return False
    heading_prefixes = (
        "section ", "coverage ", "definitions", "definition", "exclusions", "exclusion",
        "conditions", "condition", "endorsement", "schedule", "limits of liability",
    )
    if lowered.startswith(heading_prefixes):
        return True
    if re.fullmatch(r"[A-Z][A-Za-z0-9/&,\- ]{3,80}", line) and ":" not in line:
        return True
    return False


def extract_section_metadata(text: str) -> dict:
    lines = [normalize_heading(line) for line in re.split(r"\n+", text or "") if normalize_heading(line)]
    titles: List[str] = []
    for line in lines[:12]:
        if _looks_like_heading(line):
            if line not in titles:
                titles.append(line)
    section_anchor = titles[-1] if titles else None
    section_tokens = sorted(
        {
            token.lower()
            for title in titles
            for token in re.findall(r"[A-Za-z]+", title)
            if len(token) >= 3 and token.lower() not in _GENERIC_SECTION_TOKENS
        }
    )
    return {
        "section_titles": titles,
        "section_path": titles[:3],
        "section_anchor": section_anchor,
        "section_tokens": section_tokens,
        "form_codes": extract_form_codes(text or ""),
    }


def extract_section_references(text: str) -> List[str]:
    lowered = text or ""
    references: List[str] = []
    patterns = [
        r"(declarations?(?:\s+page)?)",
        r"(section\s+[ivxlcdm0-9]+(?:\s*[-:]\s*[A-Za-z][A-Za-z ]{2,})?)",
        r"(coverage\s+[a-h]\b(?:\s*[-:]\s*[A-Za-z][A-Za-z ]{2,})?)",
        r"(definitions?)",
        r"(exclusions?)",
        r"(conditions?)",
        r"(endorsement\s+[A-Z]{1,5}[- ]?\d{2,}[A-Z]?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            reference = normalize_heading(match.group(1))
            if reference and reference not in references:
                references.append(reference)
    return references


def extract_coverage_tags(text: str) -> List[str]:
    lowered = (text or "").lower()
    tags = []
    for tag, patterns in _COVERAGE_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            tags.append(tag)
    return sorted(set(tags))


def infer_document_type(source: str, text: str) -> str:
    lowered = f"{source}\n{text}".lower()
    if "declarations" in lowered or "declaration page" in lowered or "dec page" in lowered:
        return "declarations"
    if "endorsement" in lowered or "rider" in lowered or re.search(r"\bho[- ]?\d{2,}\b", lowered):
        return "endorsement"
    if "claim form" in lowered or "proof of loss" in lowered or "claim number" in lowered:
        return "claim_form"
    if "billing" in lowered or "invoice" in lowered or "amount due" in lowered or "premium due" in lowered:
        return "billing"
    if "schedule" in lowered and ("coverage" in lowered or "limit" in lowered):
        return "schedule"
    return "base_policy"


def infer_clause_types(text: str) -> List[str]:
    lowered = (text or "").lower()
    clause_types: Set[str] = set()
    if "definition" in lowered or "means" in lowered:
        clause_types.add("definition")
    if "exclusion" in lowered or "we do not cover" in lowered or "this policy excludes" in lowered:
        clause_types.add("exclusion")
    if "except" in lowered or "however" in lowered or "but this exclusion does not apply" in lowered:
        clause_types.add("exception")
    if "endorsement" in lowered or "this endorsement modifies" in lowered:
        clause_types.add("endorsement")
    if "coverage" in lowered or "we cover" in lowered or "we will pay" in lowered:
        clause_types.add("coverage")
    if "limit" in lowered or "limits of liability" in lowered or "limit of insurance" in lowered:
        clause_types.add("limit")
    if "deductible" in lowered:
        clause_types.add("deductible")
    if "premium" in lowered:
        clause_types.add("premium")
    if "insured must" in lowered or "duties after a loss" in lowered or "conditions" in lowered:
        clause_types.add("condition")
    return sorted(clause_types) or ["general"]


def primary_clause_type(text: str) -> str:
    clause_types = infer_clause_types(text)
    priority = [
        "exception",
        "endorsement",
        "definition",
        "exclusion",
        "limit",
        "deductible",
        "premium",
        "coverage",
        "condition",
        "general",
    ]
    for clause_type in priority:
        if clause_type in clause_types:
            return clause_type
    return clause_types[0]


def is_numeric_field_query(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in ["limit", "limits", "deductible", "premium", "amount", "sublimit", "retention", "coinsurance"]
    )
