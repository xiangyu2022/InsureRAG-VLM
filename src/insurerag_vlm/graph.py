from collections import defaultdict
import re
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


def _graph_group_id(record: Dict[str, Any]) -> str:
    packet_id = str(record.get("packet_id") or "").strip()
    if packet_id:
        return f"packet::{packet_id}"
    return f"doc::{record.get('doc_id')}"


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
        "relation": f"candidate_{relation}" if relation in {
            "modifies", "overridden_by", "limited_by", "qualified_by", "defines_term_for", "defines_limit_for"
        } else relation,
        "doc_id": doc_id,
        "confidence": round(confidence, 4),
        "reason": reason,
        "shared_coverages": shared_coverages or [],
        "shared_sections": shared_sections or [],
        "source_section_title": source_section_title,
        "target_section_title": target_section_title,
        "source_form_codes": source_form_codes or [],
        "relation_status": "candidate",
        "evidence_kind": "metadata_overlap",
        "confidence_kind": "heuristic_score_not_probability",
        "supports_precedence": False,
        "evidence_span": None,
    }


def build_document_graph(
    page_records: List[Dict[str, Any]],
    table_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    pages_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for page in page_records:
        pages_by_doc[_graph_group_id(page)].append(page)

    declarations_by_doc = defaultdict(list)
    endorsements_by_doc = defaultdict(list)
    definitions_by_doc = defaultdict(list)
    exclusions_by_doc = defaultdict(list)
    exceptions_by_doc = defaultdict(list)
    coverages_by_doc = defaultdict(list)
    limits_by_doc = defaultdict(list)

    for page in page_records:
        doc_id = _graph_group_id(page)
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
        if "exception" in clause_types:
            exceptions_by_doc[doc_id].append(page)
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

    for doc_id, coverage_pages in coverages_by_doc.items():
        for coverage_page in coverage_pages:
            for exclusion in exclusions_by_doc.get(doc_id, []):
                if coverage_page.get("page_key") == exclusion.get("page_key"):
                    continue
                shared_coverages = _shared_coverages(coverage_page, exclusion)
                shared_sections = _shared_sections(coverage_page, exclusion)
                if shared_coverages or shared_sections:
                    edges.append(
                        _edge(
                            coverage_page.get("page_key"),
                            exclusion.get("page_key"),
                            "limited_by",
                            doc_id,
                            confidence=0.78 if shared_coverages else 0.7,
                            reason="coverage_exclusion_overlap",
                            shared_coverages=shared_coverages,
                            shared_sections=shared_sections,
                            source_section_title=coverage_page.get("section_anchor"),
                            target_section_title=exclusion.get("section_anchor"),
                            source_form_codes=list(coverage_page.get("form_codes", []) or []),
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

    for doc_id, exception_pages in exceptions_by_doc.items():
        for exception in exception_pages:
            for exclusion in exclusions_by_doc.get(doc_id, []):
                if exception.get("page_key") == exclusion.get("page_key"):
                    continue
                shared_coverages = _shared_coverages(exception, exclusion)
                shared_sections = _shared_sections(exception, exclusion)
                if shared_coverages or shared_sections:
                    edges.append(
                        _edge(
                            exclusion.get("page_key"),
                            exception.get("page_key"),
                            "qualified_by",
                            doc_id,
                            confidence=0.8 if shared_coverages and shared_sections else 0.72,
                            reason="exception_qualifies_exclusion",
                            shared_coverages=shared_coverages,
                            shared_sections=shared_sections,
                            source_section_title=exclusion.get("section_anchor"),
                            target_section_title=exception.get("section_anchor"),
                            source_form_codes=list(exception.get("form_codes", []) or []),
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
    by_key = {p.get("page_key"): p for p in page_records}
    edges = [e for e in edges if not (
        by_key.get(e["source_page_key"], {}).get("policy_number")
        and by_key.get(e["target_page_key"], {}).get("policy_number")
        and by_key[e["source_page_key"]]["policy_number"] != by_key[e["target_page_key"]]["policy_number"])]
    edges.extend(build_explicit_reference_edges(page_records))
    edges.extend(build_annotated_reference_edges(page_records))
    return edges


def build_explicit_reference_edges(page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Resolve narrow literal section references; never infer coverage precedence.

    A heading must start the page. Repeated section IDs in a packet are ambiguous
    and produce no edge. Printed page numbers are deliberately not interpreted as
    physical PDF page numbers. Both endpoints and the literal reference are kept.
    """
    headings = defaultdict(list)
    # Preserve complete addresses: Section 5(a) cannot resolve to Section 5.
    # Unsupported suffixes fail closed instead of silently dropping detail.
    address = r"[A-Za-z0-9][A-Za-z0-9_-]{0,30}(?:\([A-Za-z0-9]{1,8}\)){0,4}"
    heading_re = re.compile(r"^\s*Section\s+(" + address + r")\s*(?::|\.(?![A-Za-z0-9]))", re.I)
    reference_re = re.compile(r"\b(?:see|refer to)\s+Section\s+(" + address
                              + r")(?![A-Za-z0-9_()\-]|\.[A-Za-z0-9]|\s+\()", re.I)
    for page in page_records:
        match = heading_re.search(str(page.get("text", "")))
        if match:
            headings[(_graph_group_id(page), match.group(1).casefold())].append((page, match.group(0).strip()))
    edges = []
    seen = set()
    for page in page_records:
        for ref in reference_re.finditer(str(page.get("text", ""))):
            targets = headings.get((_graph_group_id(page), ref.group(1).casefold()), [])
            if len(targets) != 1:
                continue
            target, heading = targets[0]
            if target["page_key"] == page["page_key"]:
                continue
            left_policy, right_policy = page.get("policy_number"), target.get("policy_number")
            if left_policy and right_policy and left_policy != right_policy:
                continue
            key = (page['page_key'], target['page_key'])
            if key in seen:
                continue
            seen.add(key)
            edge = _edge(page["page_key"], target["page_key"], "references_section", _graph_group_id(page),
                         confidence=1.0, reason="unique_literal_section_reference")
            edge.update(relation_status="explicit_reference", evidence_kind="literal_reference",
                        confidence_kind="deterministic_match_not_probability", evidence_span=ref.group(0),
                        target_evidence_span=heading, reference_id=ref.group(1),
                        evidence_page_key=page["page_key"], target_evidence_page_key=target["page_key"])
            edges.append(edge)
    return edges


def build_annotated_reference_edges(page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use source-inspected printed-to-physical page maps, with endpoint quotes.

    Mapping authorship remains explicit. Literal quote checks verify provenance,
    not the semantic correctness of a published cross-reference.
    """
    normalize = lambda value: " ".join(str(value or "").split())
    lookup = {(str(p.get("doc_id")), int(p.get("page_number", p.get("page", 0)))): p for p in page_records}
    edges = []
    for page in page_records:
        for ref in page.get("source_references", []) or []:
            target = lookup.get((str(ref.get("target_doc_id")), int(ref.get("target_physical_page", 0))))
            quote, heading = normalize(ref.get("evidence_span")), normalize(ref.get("target_evidence_span"))
            label = str(ref.get("target_printed_page_label") or "")
            valid = (target is not None and _graph_group_id(page) == _graph_group_id(target)
                     and quote and heading and quote in normalize(page.get("text"))
                     and heading in normalize(target.get("text"))
                     and str(target.get("printed_page_label") or "") == label and label.isdigit()
                     and re.search(r"\bpage\s+" + re.escape(label) + r"\b", quote, re.I))
            if not valid:
                raise ValueError("Invalid source-backed page reference: " + str(page.get("page_key")))
            if page.get("policy_number") and target.get("policy_number") and page["policy_number"] != target["policy_number"]:
                raise ValueError("Page reference crosses explicit policy identities")
            edge = _edge(page["page_key"], target["page_key"], "references_page", _graph_group_id(page),
                         confidence=1.0, reason="source_inspected_printed_to_physical_page_map")
            edge.update(relation_status="explicit_reference", evidence_kind="source_inspected_annotation",
                        confidence_kind="quote_and_mapping_check_not_probability", evidence_span=quote,
                        target_evidence_span=heading, evidence_page_key=page["page_key"],
                        target_evidence_page_key=target["page_key"], annotation_author=ref.get("annotation_author"),
                        target_printed_page_label=label, source_pdf_sha256=ref.get("source_pdf_sha256"))
            edges.append(edge)
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
    *,
    max_hops: int = 2,
    max_expansions: int = 8,
    explicit_only: bool = False,
) -> List[Dict[str, Any]]:
    if max_hops < 0 or max_expansions < 0:
        raise ValueError("Graph budgets must be nonnegative")
    if not max_hops or not max_expansions:
        return []
    allowed_relations = {"defines_limit_for", "modifies", "overridden_by", "defines_term_for", "references_section", "reverse::references_section", "references_page", "reverse::references_page"}
    if needs_declarations:
        allowed_relations.add("reverse::defines_limit_for")
    if needs_endorsement_check:
        allowed_relations.update({"modifies", "overridden_by", "reverse::modifies", "reverse::overridden_by"})
    if needs_definition:
        allowed_relations.update({"defines_term_for", "reverse::defines_term_for"})
    if needs_exclusion_review:
        allowed_relations.update(
            {
                "overridden_by",
                "reverse::overridden_by",
                "limited_by",
                "reverse::limited_by",
                "qualified_by",
                "reverse::qualified_by",
            }
        )

    expanded: List[Dict[str, Any]] = []
    visited = set(seed_page_keys)
    frontier = [(key, []) for key in sorted(seed_page_keys)]
    for hop in range(1, max_hops + 1):
        proposals = []
        for page_key, path in frontier:
            for edge in adjacency.get(page_key, []):
                relation = str(edge.get("relation"))
                base_relation = relation.replace("candidate_", "")
                explicit = edge.get("relation_status") == "explicit_reference"
                if base_relation not in allowed_relations or (explicit_only and not explicit):
                    continue
                # Heuristic links stay one-hop: transitive guesses do not become evidence.
                if hop > 1 and (not explicit or any(p.get("relation_status") != "explicit_reference" for p in path)):
                    continue
                target = str(edge.get("target_page_key"))
                if target in visited:
                    continue
                proposals.append((not explicit, -float(edge.get("confidence") or 0), target, page_key, relation, edge, path))
        next_frontier = []
        for _, _, target, page_key, relation, edge, path in sorted(proposals, key=lambda x: x[:5]):
            if target in visited:
                continue
            visited.add(target)
            step = {key: edge.get(key) for key in ("source_page_key", "target_page_key", "relation", "relation_status", "evidence_span", "target_evidence_span", "evidence_page_key", "target_evidence_page_key")}
            full_path = path + [step]
            expanded.append({**edge, "page_key": target, "source_page_key": page_key,
                             "hop": hop, "path": full_path, "supports_precedence": False})
            next_frontier.append((target, full_path))
            if len(expanded) >= max_expansions:
                return expanded
        frontier = next_frontier
        if not frontier:
            break
    return expanded
