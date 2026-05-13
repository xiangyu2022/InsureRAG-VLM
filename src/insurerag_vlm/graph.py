from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set


def _shared_coverages(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    return sorted(set(left.get("coverage_tags", []) or []) & set(right.get("coverage_tags", []) or []))


def _shared_sections(left: Dict[str, Any], right: Dict[str, Any]) -> List[str]:
    left_titles = set(left.get("section_path", []) or [])
    right_titles = set(right.get("section_path", []) or [])
    if left_titles and right_titles:
        return sorted(left_titles & right_titles)
    left_tokens = set(left.get("section_tokens", []) or [])
    right_tokens = set(right.get("section_tokens", []) or [])
    return sorted(left_tokens & right_tokens)


def _edge(
    source_page_key: str,
    target_page_key: str,
    relation: str,
    doc_id: str,
    *,
    confidence: float,
    reason: str,
    shared_coverages: List[str] | None = None,
    shared_sections: List[str] | None = None,
    source_section_title: str | None = None,
    target_section_title: str | None = None,
    source_form_codes: List[str] | None = None,
) -> Dict[str, Any]:
    return {
        "source_page_key": source_page_key,
        "target_page_key": target_page_key,
        "relation": relation,
        "doc_id": doc_id,
        "confidence": round(confidence, 4),
        "reason": reason,
        "shared_coverages": shared_coverages or [],
        "shared_sections": shared_sections or [],
        "source_section_title": source_section_title,
        "target_section_title": target_section_title,
        "source_form_codes": source_form_codes or [],
    }


def build_document_graph(
    page_records: List[Dict[str, Any]],
    table_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    pages_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for page in page_records:
        pages_by_doc[str(page.get("doc_id"))].append(page)

    declarations_by_doc = defaultdict(list)
    endorsements_by_doc = defaultdict(list)
    definitions_by_doc = defaultdict(list)
    exclusions_by_doc = defaultdict(list)
    coverages_by_doc = defaultdict(list)
    limits_by_doc = defaultdict(list)

    for page in page_records:
        doc_id = str(page.get("doc_id"))
        document_type = str(page.get("document_type", ""))
        clause_types = set(page.get("clause_types", []) or [])
        if document_type == "declarations":
            declarations_by_doc[doc_id].append(page)
        if document_type == "endorsement" or "endorsement" in clause_types:
            endorsements_by_doc[doc_id].append(page)
        if "definition" in clause_types:
            definitions_by_doc[doc_id].append(page)
        if "exclusion" in clause_types:
            exclusions_by_doc[doc_id].append(page)
        if "coverage" in clause_types:
            coverages_by_doc[doc_id].append(page)
        if "limit" in clause_types or "deductible" in clause_types or "premium" in clause_types:
            limits_by_doc[doc_id].append(page)

    for doc_id, declarations_pages in declarations_by_doc.items():
        for dec in declarations_pages:
            for page in pages_by_doc.get(doc_id, []):
                if page.get("page_key") == dec.get("page_key"):
                    continue
                shared_coverages = _shared_coverages(dec, page)
                shared_sections = _shared_sections(dec, page)
                if shared_coverages or page in limits_by_doc.get(doc_id, []):
                    edges.append(
                        _edge(
                            dec.get("page_key"),
                            page.get("page_key"),
                            "defines_limit_for",
                            doc_id,
                            confidence=0.82 if shared_coverages else 0.72,
                            reason="declarations_to_coverage_or_numeric_page",
                            shared_coverages=shared_coverages,
                            shared_sections=shared_sections,
                            source_section_title=dec.get("section_anchor"),
                            target_section_title=page.get("section_anchor"),
                            source_form_codes=list(dec.get("form_codes", []) or []),
                        )
                    )

    for doc_id, endorsement_pages in endorsements_by_doc.items():
        for endorsement in endorsement_pages:
            for page in pages_by_doc.get(doc_id, []):
                if page.get("page_key") == endorsement.get("page_key"):
                    continue
                shared_coverages = _shared_coverages(endorsement, page)
                shared_sections = _shared_sections(endorsement, page)
                target_clause_types = set(page.get("clause_types", []) or [])
                if shared_coverages or shared_sections or "exclusion" in target_clause_types:
                    confidence = 0.88 if shared_coverages and shared_sections else 0.76
                    reason = "endorsement_shared_coverage_or_section"
                    if "exclusion" in target_clause_types:
                        reason = "endorsement_targets_exclusion_or_section"
                    edges.append(
                        _edge(
                            endorsement.get("page_key"),
                            page.get("page_key"),
                            "modifies",
                            doc_id,
                            confidence=confidence,
                            reason=reason,
                            shared_coverages=shared_coverages,
                            shared_sections=shared_sections,
                            source_section_title=endorsement.get("section_anchor"),
                            target_section_title=page.get("section_anchor"),
                            source_form_codes=list(endorsement.get("form_codes", []) or []),
                        )
                    )

    for doc_id, exclusion_pages in exclusions_by_doc.items():
        for exclusion in exclusion_pages:
            for endorsement in endorsements_by_doc.get(doc_id, []):
                shared_coverages = _shared_coverages(exclusion, endorsement)
                shared_sections = _shared_sections(exclusion, endorsement)
                if shared_coverages or shared_sections:
                    edges.append(
                        _edge(
                            exclusion.get("page_key"),
                            endorsement.get("page_key"),
                            "overridden_by",
                            doc_id,
                            confidence=0.9 if shared_coverages else 0.78,
                            reason="exclusion_and_endorsement_overlap",
                            shared_coverages=shared_coverages,
                            shared_sections=shared_sections,
                            source_section_title=exclusion.get("section_anchor"),
                            target_section_title=endorsement.get("section_anchor"),
                            source_form_codes=list(endorsement.get("form_codes", []) or []),
                        )
                    )

    for doc_id, definition_pages in definitions_by_doc.items():
        for definition in definition_pages:
            for page in pages_by_doc.get(doc_id, []):
                if page.get("page_key") == definition.get("page_key"):
                    continue
                shared_coverages = _shared_coverages(definition, page)
                shared_sections = _shared_sections(definition, page)
                if shared_coverages or shared_sections:
                    edges.append(
                        _edge(
                            definition.get("page_key"),
                            page.get("page_key"),
                            "defines_term_for",
                            doc_id,
                            confidence=0.74,
                            reason="definition_overlap",
                            shared_coverages=shared_coverages,
                            shared_sections=shared_sections,
                            source_section_title=definition.get("section_anchor"),
                            target_section_title=page.get("section_anchor"),
                            source_form_codes=list(definition.get("form_codes", []) or []),
                        )
                    )

    for table_record in table_records:
        field_type = str(table_record.get("field_type", ""))
        if field_type in {"limit", "deductible", "premium"}:
            edges.append(
                _edge(
                    table_record.get("page_key"),
                    table_record.get("page_key"),
                    f"table_{field_type}",
                    str(table_record.get("doc_id")),
                    confidence=0.7,
                    reason="normalized_table_field",
                    shared_coverages=list(table_record.get("coverage_tags", []) or []),
                    shared_sections=list(table_record.get("section_path", []) or []),
                    source_section_title=table_record.get("section_anchor"),
                    target_section_title=table_record.get("section_anchor"),
                    source_form_codes=list(table_record.get("form_codes", []) or []),
                )
            )
    return edges


def build_graph_adjacency(edges: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    adjacency: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        adjacency[str(edge.get("source_page_key"))].append(edge)
        adjacency[str(edge.get("target_page_key"))].append(
            {
                **edge,
                "source_page_key": edge.get("target_page_key"),
                "target_page_key": edge.get("source_page_key"),
                "relation": f"reverse::{edge.get('relation')}",
            }
        )
    return adjacency


def expand_candidate_page_keys(
    seed_page_keys: Set[str],
    adjacency: Dict[str, List[Dict[str, Any]]],
    needs_endorsement_check: bool,
    needs_declarations: bool,
    needs_definition: bool,
    needs_exclusion_review: bool,
) -> List[Dict[str, Any]]:
    allowed_relations = {"defines_limit_for", "modifies", "overridden_by", "defines_term_for"}
    if needs_declarations:
        allowed_relations.add("reverse::defines_limit_for")
    if needs_endorsement_check:
        allowed_relations.update({"modifies", "overridden_by", "reverse::modifies", "reverse::overridden_by"})
    if needs_definition:
        allowed_relations.update({"defines_term_for", "reverse::defines_term_for"})
    if needs_exclusion_review:
        allowed_relations.update({"overridden_by", "reverse::overridden_by"})

    expanded: List[Dict[str, Any]] = []
    seen_targets: Set[str] = set()
    for page_key in seed_page_keys:
        for edge in adjacency.get(page_key, []):
            relation = str(edge.get("relation"))
            if relation not in allowed_relations:
                continue
            target_page_key = str(edge.get("target_page_key"))
            if target_page_key in seed_page_keys or target_page_key in seen_targets:
                continue
            seen_targets.add(target_page_key)
            expanded.append(
                {
                    "page_key": target_page_key,
                    "relation": relation,
                    "source_page_key": page_key,
                    "confidence": edge.get("confidence"),
                    "reason": edge.get("reason"),
                    "shared_coverages": edge.get("shared_coverages", []),
                    "shared_sections": edge.get("shared_sections", []),
                    "source_section_title": edge.get("source_section_title"),
                    "target_section_title": edge.get("target_section_title"),
                    "source_form_codes": edge.get("source_form_codes", []),
                }
            )
    return expanded
