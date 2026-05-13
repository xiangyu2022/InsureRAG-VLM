import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .insurance_structure import extract_coverage_tags, extract_section_references, is_numeric_field_query


@dataclass
class QueryUnderstanding:
    intent: str
    target_coverages: List[str]
    needs_limit: bool
    needs_endorsement_check: bool
    needs_table_lookup: bool
    needs_definition: bool
    needs_exclusion_review: bool
    needs_declarations: bool
    needs_graph_expansion: bool
    preferred_document_types: List[str] = field(default_factory=list)
    preferred_clause_types: List[str] = field(default_factory=list)
    preferred_field_types: List[str] = field(default_factory=list)
    preferred_sections: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def understand_query(question: str) -> QueryUnderstanding:
    lowered = (question or "").lower()
    target_coverages = extract_coverage_tags(question)
    preferred_sections = extract_section_references(question)

    if re.search(r"\b(compare|difference|different|drift|changed?|version)\b", lowered):
        intent = "policy_diff"
    elif re.search(r"\bdefine|definition|mean|what is\b", lowered):
        intent = "definition_lookup"
    elif re.search(r"\bcovered|coverage|cover|apply|included|excluded|exclusion\b", lowered):
        intent = "coverage_check"
    elif is_numeric_field_query(question):
        intent = "limit_lookup"
    else:
        intent = "document_qa"

    needs_limit = is_numeric_field_query(question)
    needs_definition = intent == "definition_lookup" or "defined" in lowered
    needs_exclusion_review = any(term in lowered for term in ["exclusion", "excluded", "not covered", "exception"])
    needs_endorsement_check = any(
        term in lowered
        for term in ["endorsement", "rider", "override", "modify", "modifies", "added back", "add back"]
    )
    needs_declarations = any(
        term in lowered
        for term in ["declarations", "declaration page", "premium", "deductible", "policy period"]
    ) or needs_limit
    needs_table_lookup = any(
        term in lowered
        for term in ["schedule", "table", "premium", "deductible", "limit", "limits", "amount"]
    )
    needs_graph_expansion = needs_endorsement_check or needs_declarations or needs_definition or needs_exclusion_review
    preferred_document_types: List[str] = []
    preferred_clause_types: List[str] = []
    preferred_field_types: List[str] = []

    if needs_declarations:
        preferred_document_types.append("declarations")
    if needs_endorsement_check:
        preferred_document_types.append("endorsement")
        preferred_clause_types.append("endorsement")
    if needs_definition:
        preferred_clause_types.append("definition")
    if needs_exclusion_review:
        preferred_clause_types.append("exclusion")
    if intent == "coverage_check":
        preferred_clause_types.append("coverage")

    if "deductible" in lowered:
        preferred_field_types.append("deductible")
        preferred_clause_types.append("deductible")
    if "premium" in lowered:
        preferred_field_types.append("premium")
        preferred_clause_types.append("premium")
    if any(term in lowered for term in ["limit", "limits", "sublimit", "retention", "coinsurance"]):
        preferred_field_types.append("limit")
        preferred_clause_types.append("limit")
    if needs_table_lookup and "schedule" in lowered:
        preferred_document_types.append("schedule")

    preferred_document_types = sorted(set(preferred_document_types))
    preferred_clause_types = sorted(set(preferred_clause_types))
    preferred_field_types = sorted(set(preferred_field_types))

    return QueryUnderstanding(
        intent=intent,
        target_coverages=target_coverages,
        needs_limit=needs_limit,
        needs_endorsement_check=needs_endorsement_check,
        needs_table_lookup=needs_table_lookup,
        needs_definition=needs_definition,
        needs_exclusion_review=needs_exclusion_review,
        needs_declarations=needs_declarations,
        needs_graph_expansion=needs_graph_expansion,
        preferred_document_types=preferred_document_types,
        preferred_clause_types=preferred_clause_types,
        preferred_field_types=preferred_field_types,
        preferred_sections=preferred_sections,
    )
