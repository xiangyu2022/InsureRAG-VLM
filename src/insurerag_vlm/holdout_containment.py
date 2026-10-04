"""Exact evidence-window screening inside arbitrarily long historical records.

Shared wording is a review flag, not proof that two information needs coincide.
No-match results are not semantic, source-family, or pretraining clearance.
"""
from collections import defaultdict
from hashlib import sha256
import unicodedata

from .holdout_quality import normalize


def verify_evidence_projection(rows, documents):
    """Bind projected text to immutable source offsets before calling it evidence."""
    checked_documents=set()
    for row in rows:
        span=row.get('evidence')
        if not row.get('parent_id') or not isinstance(span,dict):
            raise ValueError('Evidence projection requires parent_id and source span')
        document=documents.get(span.get('document_id'))
        if document is None:raise ValueError('Unknown evidence document')
        text=document['text']
        if span['document_id'] not in checked_documents:
            if sha256(text.encode()).hexdigest()!=document.get('normalized_text_sha256'):
                raise ValueError('Source document hash mismatch')
            checked_documents.add(span['document_id'])
        start,end=span.get('start'),span.get('end')
        if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text):
            raise ValueError('Invalid evidence offsets')
        original=text[start:end]
        if sha256(original.encode()).hexdigest()!=span.get('sha256') or row.get('answer')!=original:
            raise ValueError('Projected evidence does not match source span')
    return len(rows)


def evidence_windows(rows, field='answer', window_words=16, minimum_words=8):
    if not 1 <= minimum_words <= window_words:
        raise ValueError('Require 1 <= minimum_words <= window_words')
    index = defaultdict(list)
    seen_ids = set()
    for row in rows:
        ident = row['id']
        if ident in seen_ids:
            raise ValueError('Duplicate candidate id: ' + str(ident))
        seen_ids.add(ident)
        raw = row[field]
        if not isinstance(raw, str):
            raise ValueError('Candidate evidence must be text')
        variants = {normalize(raw), normalize(''.join(
            char for char in raw if unicodedata.category(char) != 'Cf'))}
        for variant in sorted(variants):
            tokens = variant.split()
            width = min(window_words, len(tokens))
            if width < minimum_words:
                continue
            variant_hash = sha256(variant.encode()).hexdigest()
            for start in range(len(tokens) - width + 1):
                window = tuple(tokens[start:start + width])
                index[window].append({
                    'candidate_id': ident,
                    'variant_sha256': variant_hash,
                    'start_token': start,
                    'end_token': start + width,
                    'evidence_tokens': len(tokens),
                })
    return dict(index)


def scan_history(index, historical_rows, progress=None):
    """Stream records independently; keep first matching origin per window."""
    remaining = defaultdict(lambda: defaultdict(set))
    for window in index:
        remaining[window[0]][len(window)].add(window)
    matches = []
    covered = defaultdict(set)
    lengths = {}
    count = 0
    for row in historical_rows:
        count += 1
        text = row['text']
        tokens = normalize(text).split()
        text_hash = None
        for start, token in enumerate(tokens):
            groups = remaining.get(token)
            if not groups:
                continue
            for width, choices in groups.items():
                if not choices:
                    continue
                window = tuple(tokens[start:start + width])
                if window not in choices:
                    continue
                choices.remove(window)
                if text_hash is None:
                    text_hash = sha256(text.encode()).hexdigest()
                locations = index[window]
                matches.append({
                    'window': ' '.join(window),
                    'origin': row.get('origin'),
                    'historical_text_sha256': text_hash,
                    'historical_token_start': start,
                    'candidate_locations': locations,
                })
                for location in locations:
                    key = (location['candidate_id'], location['variant_sha256'])
                    lengths[key] = location['evidence_tokens']
                    covered[key].update(range(location['start_token'], location['end_token']))
        if progress and count % 100000 == 0:
            progress(count, len(matches))
    coverage = [
        {'candidate_id': key[0], 'variant_sha256': key[1],
         'matched_tokens_across_history': len(positions),
         'evidence_tokens': lengths[key],
         'token_coverage_across_history': len(positions) / lengths[key]}
        for key, positions in sorted(covered.items())
    ]
    return {'historical_strings': count, 'unique_windows': len(index),
            'matched_windows': len(matches),
            'flagged_candidate_ids': sorted({key[0] for key in covered}),
            'coverage': coverage, 'matches': matches}
