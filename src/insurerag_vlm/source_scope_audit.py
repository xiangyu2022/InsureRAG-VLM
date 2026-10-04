"""Research-only warnings from explicit source metadata.

This module does not infer jurisdiction from prose, a domain suffix or a
program name. It does not rank, filter, validate claims or authorize an answer.
It is deliberately not wired into the frozen inference pipeline.
"""


def _scope(value):
    if value is not None and not isinstance(value,str):
        raise ValueError('Jurisdiction metadata must be text or null')
    if value is None or not value.strip():
        return None
    return str(value).strip().upper()


def audit_source_scope(sources, expected_jurisdiction=None):
    """Compare explicit opaque scope labels; no geographic hierarchy assumed.

    A caller may use labels such as IE, CA-ON or a policy-specific authority.
    Different label granularity is reported for review, never resolved by
    assuming one source applies to another jurisdiction.
    """
    expected=_scope(expected_jurisdiction)
    records=[]
    for source in sources:
        if not isinstance(source,dict) or not isinstance(source.get('source'),str) or not source['source'].strip():
            raise ValueError('Every source requires an explicit nonempty ID')
        if source['source']!=source['source'].strip():raise ValueError('Source IDs must be canonical without surrounding whitespace')
        if source.get('publisher') is not None and not isinstance(source['publisher'],str):
            raise ValueError('Publisher metadata must be text or null')
        records.append({'source':str(source['source']),
                        'jurisdiction':_scope(source.get('jurisdiction')),
                        'publisher':(source.get('publisher') or '').strip() or None})
    if len({r['source'] for r in records})!=len(records):
        raise ValueError('Duplicate source IDs make scope attribution ambiguous')
    scopes=sorted({r['jurisdiction'] for r in records if r['jurisdiction']})
    unknown=[r['source'] for r in records if r['jurisdiction'] is None]
    unknown_publisher=[r['source'] for r in records if r['publisher'] is None]
    mismatch=[r['source'] for r in records if expected and r['jurisdiction'] and r['jurisdiction']!=expected]
    flags=[]
    if not records:flags.append('no_sources')
    if unknown:flags.append('jurisdiction_metadata_missing')
    if unknown_publisher:flags.append('publisher_metadata_missing')
    if len(scopes)>1:flags.append('multiple_declared_jurisdictions')
    if mismatch:flags.append('declared_scope_differs_from_requested_scope')
    if expected is None:flags.append('requested_jurisdiction_not_explicit')
    return {'mode':'research_warning_only','source_count':len(records),
            'expected_jurisdiction':expected,'declared_jurisdictions':scopes,
            'unknown_jurisdiction_source_ids':unknown,'mismatched_source_ids':mismatch,
            'unknown_publisher_source_ids':unknown_publisher,
            'flags':flags,'semantic_support':'not_assessed','serving_action':'none',
            'limitations':'Metadata may be missing or wrong; matching scope does not prove relevance, entailment, currency or personal-policy applicability.'}
