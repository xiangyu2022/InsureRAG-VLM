import re
from typing import Any, Dict, List, Optional

from .insurance_structure import extract_coverage_tags


_AMOUNT_RE = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d+(?:\.\d+)?%")
_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")


def _normalize_field_name(field_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", (field_name or "").lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_field_value(field_value: str) -> str:
    normalized = re.sub(r"\s+", " ", field_value or "").strip()
    normalized = normalized.replace(" ,", ",")
    return normalized


def _field_type(field_name: str, row_text: str) -> str:
    lowered = f"{field_name} {row_text}".lower()
    if "deductible" in lowered:
        return "deductible"
    if "premium" in lowered:
        return "premium"
    if "limit" in lowered or "limits" in lowered or "sublimit" in lowered or "retention" in lowered:
        return "limit"
    if "effective" in lowered or "expiration" in lowered or "policy period" in lowered:
        return "policy_period"
    if "endorsement" in lowered:
        return "endorsement"
    return "table_value"


def _parse_numeric_value(field_value: str) -> tuple[Optional[float], Optional[str]]:
    value = field_value or ""
    if not value:
        return None, None
    amount_match = _AMOUNT_RE.search(value)
    if not amount_match:
        return None, None
    token = amount_match.group(0)
    if token.endswith("%"):
        try:
            return float(token.rstrip("%").replace(",", "")), "percent"
        except ValueError:
            return None, "percent"
    try:
        return float(token.replace("$", "").replace(",", "")), "currency"
    except ValueError:
        return None, "currency"


def _table_lines(text: str) -> List[str]:
    return [line.strip() for line in re.split(r"\n+", text or "") if line.strip()]


def _table_coverage_tags(line: str, page_record: Dict[str, Any]) -> List[str]:
    tags = extract_coverage_tags(line)
    if tags:
        return tags
    return list(page_record.get("coverage_tags", []) or [])


def _build_table_record(
    page_record: Dict[str, Any],
    idx: int,
    field_name: str,
    field_value: str,
    line: str,
) -> Dict[str, Any]:
    normalized_field_name = _normalize_field_name(field_name)
    normalized_field_value = _normalize_field_value(field_value)
    numeric_value, value_unit = _parse_numeric_value(normalized_field_value)
    return {
        "record_id": f"{page_record.get('page_key')}::table::{idx:03d}",
        "record_type": "table",
        "doc_id": page_record.get("doc_id"),
        "page_key": page_record.get("page_key"),
        "parent_page_id": page_record.get("page_key"),
        "source": page_record.get("source"),
        "page_number": page_record.get("page_number"),
        "field_name": field_name,
        "field_value": field_value,
        "normalized_field_name": normalized_field_name,
        "normalized_field_value": normalized_field_value,
        "field_type": _field_type(field_name, line),
        "numeric_value": numeric_value,
        "value_unit": value_unit,
        "text": line,
        "coverage_tags": _table_coverage_tags(line, page_record),
        "document_type": page_record.get("document_type"),
        "section_anchor": page_record.get("section_anchor"),
        "section_path": list(page_record.get("section_path", []) or []),
        "section_titles": list(page_record.get("section_titles", []) or []),
        "section_tokens": list(page_record.get("section_tokens", []) or []),
        "form_codes": list(page_record.get("form_codes", []) or []),
    }


def extract_table_records_from_page(page_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    lines = _table_lines(str(page_record.get("text", "")))

    for idx, line in enumerate(lines):
        if ":" in line:
            left, right = [part.strip() for part in line.split(":", 1)]
            if right and (_AMOUNT_RE.search(right) or _DATE_RE.search(right) or len(right) <= 80):
                records.append(_build_table_record(page_record, idx, left, right, line))
                continue
        if "|" in line:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if len(cells) >= 2:
                field_name = cells[0]
                field_value = " | ".join(cells[1:])
                records.append(_build_table_record(page_record, idx, field_name, field_value, line))
                continue
        if _AMOUNT_RE.search(line) and len(line) <= 160:
            field_name = line.split("$", 1)[0].strip() or "value"
            field_value = " ".join(_AMOUNT_RE.findall(line))
            records.append(_build_table_record(page_record, idx, field_name, field_value, line))
    return records


def build_table_records(page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for page_record in page_records:
        records.extend(extract_table_records_from_page(page_record))
    return records


def serialize_table_record(record: Dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in [
            record.get("field_name", ""),
            record.get("normalized_field_name", ""),
            record.get("field_value", ""),
            record.get("normalized_field_value", ""),
            record.get("field_type", ""),
            record.get("section_anchor", ""),
            " ".join(record.get("coverage_tags", []) or []),
            record.get("text", ""),
        ]
        if value
    )
