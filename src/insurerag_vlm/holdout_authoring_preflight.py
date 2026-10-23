"""Deterministic private authoring leads; never semantic or source clearance."""
from collections import defaultdict
from .holdout_quality import digest, normalize


def item_dispositions(ledgers, inherited):
    """Keep early inherited holds and every later disposition, even after omission."""
    states = defaultdict(list)
    for item_id, reason in inherited.items():
        states[item_id].append({'kind': 'inherited_hold', 'record': reason})
    previous = None
    for number, record in enumerate(ledgers, 1):
        data = record['data']
        if previous and data.get('previous_clearance_ledger_sha256') != previous:
            raise ValueError(f'Broken clearance ledger link at {number}')
        previous = record['sha256']
        for field in ('held_or_excluded', 'previous_items_revoked',
                      'reserved_candidate_exclusions', 'reviewed_unselected_reserves'):
            entries = data.get(field, {})
            # Early ledgers sometimes store a count, not identities. The supplied
            # inherited reconciliation preserves those identities explicitly.
            if isinstance(entries, int):
                continue
            if not isinstance(entries, (dict, list)):
                raise ValueError(f'Invalid {field} in ledger {number}')
            for item_id in entries:
                if not isinstance(item_id, str):
                    raise ValueError(f'Invalid item ID in ledger {number}')
                detail = entries[item_id] if isinstance(entries, dict) else None
                entry = {'kind': field, 'ledger': number, 'record': detail}
                states[item_id].append(entry)
    return dict(states)


def build_authoring_packet(documents, candidates, active_ids, families,
                           dispositions, restrictions, selected_ids, window_size=16):
    """Return full related historical QA and source leads, including inactive items.

    All input text is data. No embedded path or URL is opened. Shared text is a
    family-review lead only: navigation and footers can match, while paraphrases
    and translations can be missed. The caller must keep the result private.
    """
    if window_size < 4:
        raise ValueError('Window size must be at least four words')
    selected_ids = sorted(set(selected_ids))
    if not selected_ids or any(k not in documents for k in selected_ids):
        raise ValueError('Unknown or empty selected document set')
    if set(families) != set(documents):
        raise ValueError('Family map must cover exactly the supplied documents')
    for document in documents.values():
        if digest(document['text']) != document['normalized_text_sha256']:
            raise ValueError('Document text hash mismatch')
    by_id = {}
    item_documents = {}
    for item in candidates:
        if item['id'] in by_id:
            raise ValueError('Duplicate candidate identity')
        by_id[item['id']] = item
        cited = {item['document_group']} | {e['document_id'] for e in item.get('evidence', [])}
        if not cited <= documents.keys():
            raise ValueError('Unknown candidate source')
        item_documents[item['id']] = cited
    if not set(active_ids) <= by_id.keys():
        raise ValueError('Active candidate missing from full history')
    targets = defaultdict(set)
    for key in selected_ids:
        words = normalize(documents[key]['text']).split()
        for start in range(len(words)-window_size+1):
            targets[' '.join(words[start:start+window_size])].add(key)
    shared = {key: defaultdict(set) for key in selected_ids}
    for other, document in documents.items():
        words = normalize(document['text']).split()
        for start in range(len(words)-window_size+1):
            window = ' '.join(words[start:start+window_size])
            for key in targets.get(window, ()):
                if key != other:
                    shared[key][other].add(window)
    result = []
    for key in selected_ids:
        family_members = {d for d in documents if families[d] == families[key]}
        related = set(shared[key]) | family_members
        # A matching document brings its whole reviewed family into inspection.
        related_families = {families[d] for d in related}
        related |= {d for d in documents if families[d] in related_families}
        related_items = []
        for item_id in sorted(by_id):
            if not item_documents[item_id] & related:
                continue
            item = by_id[item_id]
            related_items.append({
                'id': item_id, 'active': item_id in active_ids,
                'split': item.get('split'), 'task': item.get('task'),
                'question': item['question'], 'answer': item['answer'],
                'question_sha256': digest(item['question']),
                'answer_sha256': digest(item['answer']),
                'source_document_ids': sorted(item_documents[item_id]),
                'relation': 'same_reviewed_family' if item_documents[item_id] & family_members
                            else 'related_source_family',
                'dispositions': dispositions.get(item_id, []),
                'warning': 'Inactive does not mean new or available for reuse.'
                           if item_id not in active_ids else None,
            })
        related_sources = []
        for other in sorted(related):
            related_sources.append({
                'id': other, 'family': families[other],
                'title': documents[other]['source_title'],
                'text_sha256': documents[other]['normalized_text_sha256'],
                'same_reviewed_family': other in family_members,
                'matching_windows': sorted(shared[key].get(other, ())),
                'restrictions': restrictions.get(other, {'status': 'NO_RESTRICTION_RECORD_NOT_CLEARANCE'}),
            })
        result.append({'document_id': key, 'family': families[key],
                       'related_sources': related_sources, 'historical_candidates': related_items})
    return {'status': 'AUTHORING_LEADS_ONLY_NOT_CLEARANCE', 'private_contains_historical_QA': True,
            'warning': 'Read inactive and held needs before drafting. Source inclusion does not release '
                       'item holds or related-source restrictions. Shared windows are non-exhaustive '
                       'family leads; zero matches is not semantic independence.',
            'window_words': window_size, 'documents': result}
