"""Source-only deterministic chunks and exact-span mapping, not semantic clearance.

The caller supplies a separately reviewed inclusion manifest and a pinned local
encoder token counter. Question text and answer labels are never inputs to
chunk_document. Atomic ranges must include table headers and their qualifiers;
oversized atomic text fails rather than silently dropping rows or truncating it.
"""
import hashlib
import json
import re


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def _ranges(values, length):
    result = []
    for pair in values:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(x) is not int for x in pair)):
            raise ValueError('Ranges require two integer offsets')
        a, b = pair
        if not 0 <= a < b <= length or (result and a < result[-1][1]):
            raise ValueError('Ranges must be ordered, disjoint and within source')
        result.append((a, b))
    return result


def chunk_document(document, inclusion, token_count, *, max_tokens=512, max_chars=2400):
    """Partition every included source character; never silently trim evidence.

    token_count must count exactly what the pinned document encoder receives,
    including special tokens. This function does not load or download a model.
    A manifest is an auditable review assertion, not independent proof of rights.
    """
    if any(type(n) is not int or n < 1 for n in (max_tokens, max_chars)):
        raise ValueError('Positive integer budgets required')
    text = document['text']
    if (inclusion.get('document_id') != document['id']
            or inclusion.get('source_sha256') != document['source_sha256']
            or inclusion.get('text_sha256') != digest(text)
            or document['normalized_text_sha256'] != digest(text)):
        raise ValueError('Source identity or hash mismatch')
    review = inclusion.get('review', {})
    if (review.get('decision') != 'include' or not review.get('scope_rationale')
            or not re.fullmatch('[0-9a-f]{64}', review.get('record_sha256', ''))
            or inclusion.get('selection_basis') != 'source_only'
            or inclusion.get('atomic_ranges_reviewed') is not True):
        raise ValueError('Explicit source-only rights/scope/structure review required')
    spans = _ranges(inclusion.get('ranges', []), len(text))
    atoms = _ranges(inclusion.get('atomic_ranges', []), len(text))
    if not spans or any(not any(a <= x < y <= b for a,b in spans) for x,y in atoms):
        raise ValueError('Missing inclusion or partially included atomic range')
    def fits(part):
        count = token_count(part)
        if type(count) is not int or count < 1:
            raise ValueError('Token counter must return positive integer')
        return len(part) <= max_chars and count <= max_tokens
    for a,b in atoms:
        if not fits(text[a:b]):
            raise ValueError('Atomic source range exceeds encoder/context budget')
    manifest_hash = digest(json.dumps(inclusion, sort_keys=True, separators=(',', ':')))
    chunks = []
    for start, stop in spans:
        # Whitespace boundaries preserve exact source characters. An atomic
        # range is never split, including when it crosses a natural boundary.
        ends = {start + m.end() for m in re.finditer(r'\S+\s*', text[start:stop])}
        ends.add(stop)
        ends = sorted(e for e in ends if not any(a < e < b for a,b in atoms))
        cursor = start
        while cursor < stop:
            available = [e for e in ends if cursor < e <= cursor + max_chars]
            chosen = None
            for end in reversed(available):
                if fits(text[cursor:end]):
                    chosen = end
                    break
            if chosen is None or not text[cursor:chosen].strip():
                raise ValueError('Source cannot fit without dropping or splitting protected text')
            part = text[cursor:chosen]
            identity = json.dumps([document['id'], digest(text), cursor, chosen, digest(part)], separators=(',', ':'))
            chunks.append(dict(id=digest(identity), document_id=document['id'],
                               start=cursor, end=chosen, text=part, sha256=digest(part),
                               source_sha256=document['source_sha256'], text_sha256=digest(text),
                               inclusion_sha256=manifest_hash, tokens=token_count(part)))
            cursor = chosen
    return chunks


def map_support_span(evidence, document, chunks):
    """Map exact source support to all required chunks; gaps fail closed.

    This proves offset coverage only. The caller still reviews support-unit
    necessity and equivalent aliases. Refusal spans are gap-context diagnostics,
    never answer-relevance gold.
    """
    a, b = _ranges([[evidence['start'], evidence['end']]], len(document['text']))[0]
    if (evidence['document_id'] != document['id']
            or evidence['sha256'] != digest(document['text'][a:b])
            or document['normalized_text_sha256'] != digest(document['text'])):
        raise ValueError('Evidence source/hash mismatch')
    relevant = []
    seen = set()
    for c in chunks:
        if c['id'] in seen:
            raise ValueError('Duplicate chunk ID')
        seen.add(c['id'])
        if c['document_id'] != document['id']:
            continue
        x,y = _ranges([[c['start'],c['end']]],len(document['text']))[0]
        identity = json.dumps([document['id'], digest(document['text']), x, y, digest(c['text'])], separators=(',', ':'))
        if c['id'] != digest(identity):
            raise ValueError('Chunk identity mismatch')
        if (c['text_sha256'] != digest(document['text'])
                or c['source_sha256'] != document['source_sha256']
                or c['text'] != document['text'][x:y] or c['sha256'] != digest(c['text'])):
            raise ValueError('Chunk provenance mismatch')
        if x < b and y > a:
            relevant.append(c)
    relevant.sort(key=lambda c: (c['start'], c['end'], c['id']))
    cursor = a
    for c in relevant:
        if c['start'] > cursor:
            raise ValueError('Evidence has uncovered source characters')
        if c['start'] < cursor and cursor != a:
            raise ValueError('Overlapping chunks need explicit alternative annotations')
        cursor = max(cursor, c['end'])
    if cursor < b:
        raise ValueError('Evidence has uncovered source characters')
    return [c['id'] for c in relevant]
