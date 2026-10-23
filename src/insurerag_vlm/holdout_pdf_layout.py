"""Render explicitly reviewed PDF word order without losing source words.

This module does not infer reading order, table meaning, rights, or source quality.
A source-only layout is a review assertion bound to the complete extracted word
inventory and original PDF hash. Every word must occur exactly once. Raw PDFs and
prior canonical extractions stay immutable; callers review new text and rebind
evidence before admitting any replacement snapshot.
"""
import hashlib
import json
import math
import re


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def render_reviewed_words(pages, layout, *, source_sha256):
    """Return literal word text, page/group offsets, and geometric provenance.

    ``pages`` contains page_number, width, height and words (text, bbox).
    ``layout.pages`` specifies ordered groups of word_indices for every page.
    Group IDs/roles are metadata, never injected into source text. Separators are
    the only new characters. Complete-word coverage is mechanical, not proof that
    a reviewer assigned the correct reading order or table relationships.
    """
    if (not _hash(source_sha256) or not isinstance(layout, dict)
            or set(layout) != {'source_sha256', 'status', 'qa_inputs_used',
                               'review_record_sha256', 'word_pages_sha256', 'pages'}
            or layout.get('source_sha256') != source_sha256
            or layout.get('status') != 'REVIEWED_SOURCE_ONLY_LAYOUT'
            or layout.get('qa_inputs_used') is not False
            or not _hash(layout.get('review_record_sha256'))):
        raise ValueError('Hash-bound source-only layout review required')
    if not isinstance(pages, list) or not pages:
        raise ValueError('Complete page inventory required')
    if layout.get('word_pages_sha256') != fingerprint(pages):
        raise ValueError('Source word inventory hash mismatch')
    plans = layout.get('pages')
    if not isinstance(plans, list) or len(plans) != len(pages):
        raise ValueError('Layout must cover every source page')
    parts = []; page_spans = []; group_spans = []; word_spans = []; cursor = 0
    for number, (page, plan) in enumerate(zip(pages, plans), 1):
        if (not isinstance(page, dict) or not isinstance(plan, dict)
                or set(page) != {'page_number', 'width', 'height', 'words'}
                or set(plan) != {'page_number', 'groups'}
                or type(page.get('page_number')) is not int
                or type(plan.get('page_number')) is not int
                or page['page_number'] != number or plan['page_number'] != number):
            raise ValueError('Pages must be complete, unique and ordered')
        width, height = page.get('width'), page.get('height')
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
               for x in (width, height)):
            raise ValueError('Finite positive page dimensions required')
        words = page.get('words'); groups = plan.get('groups')
        if not isinstance(words, list) or not isinstance(groups, list):
            raise ValueError('Word and group inventories must be lists')
        for word in words:
            if (not isinstance(word, dict) or set(word) != {'text', 'bbox'}
                    or not isinstance(word.get('text'), str)
                    or not word['text'].strip()):
                raise ValueError('Source word text cannot be empty')
            box = word.get('bbox')
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError('Source word geometry is outside its page')
        seen = set(); group_ids = set(); page_start = cursor
        for group_number, group in enumerate(groups):
            if (not isinstance(group, dict) or set(group) != {'id', 'role', 'word_indices'}
                    or not isinstance(group['id'], str) or not group['id'].strip()
                    or group['id'] in group_ids
                    or not isinstance(group['role'], str)
                    or group['role'] not in {'heading', 'paragraph', 'table_header', 'table_cell',
                                             'table_note', 'sidebar', 'footer', 'other'}):
                raise ValueError('Unique reviewed group identity and supported role required')
            group_ids.add(group['id']); indices = group['word_indices']
            if not isinstance(indices, list) or not indices:
                raise ValueError('Reviewed group must contain source words')
            if group_number:
                parts.append('\n\n'); cursor += 2
            start = cursor
            for position, index in enumerate(indices):
                if type(index) is not int or not 0 <= index < len(words) or index in seen:
                    raise ValueError('Word assignment is duplicate or outside the source inventory')
                seen.add(index)
                if position:
                    parts.append(' '); cursor += 1
                word = words[index]; text = word['text']; parts.append(text)
                word_spans.append(dict(page_number=number, word_index=index, start=cursor,
                                       end=cursor+len(text), bbox=list(word['bbox']),
                                       sha256=hashlib.sha256(text.encode('utf8')).hexdigest()))
                cursor += len(text)
            group_spans.append(dict(page_number=number, group_id=group['id'], role=group['role'],
                                    start=start, end=cursor, word_indices=list(indices)))
        if seen != set(range(len(words))):
            raise ValueError('Layout omits source words')
        page_spans.append(dict(page_number=number, start=page_start, end=cursor))
        if number < len(pages):
            parts.append('\n\f\n'); cursor += 3
    text = ''.join(parts); assert cursor == len(text)
    for entry in page_spans + group_spans:
        entry['sha256'] = hashlib.sha256(text[entry['start']:entry['end']].encode('utf8')).hexdigest()
    return dict(text=text, normalized_text_sha256=hashlib.sha256(text.encode('utf8')).hexdigest(),
                page_spans=page_spans, group_spans=group_spans, word_spans=word_spans,
                source_sha256=source_sha256, word_pages_sha256=fingerprint(pages),
                layout_sha256=fingerprint(layout), all_source_words_preserved=True,
                source_quality_or_rights_cleared=False, evidence_rebinding_required=True)
