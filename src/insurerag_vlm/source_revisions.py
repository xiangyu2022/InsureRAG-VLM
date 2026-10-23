"""Faithful text transcription of explicitly reviewed source deletions.

This preserves deleted words as labeled metadata rather than silently choosing
a historical or current rule. Source scope and rights require separate review.
"""
import copy
import re

from bs4 import NavigableString


DELETION_PREFIX = '[deleted in source: '


def transcribe_reviewed_deletions(root, selectors):
    """Return normalized text and original deletion metadata without mutation.

    Only explicitly selected, nonempty plain-text deletion nodes are supported.
    A span must have an inline text-decoration containing line-through; del,
    strike and s tags carry their own deletion markup. Unselected deletion
    nodes are rejected so that no source deletion silently becomes ordinary text.
    """
    if not isinstance(selectors, (list, tuple)) or not selectors:
        raise ValueError('Explicit reviewed selectors required')
    if DELETION_PREFIX in root.get_text():
        raise ValueError('Source collides with deletion metadata marker')
    work = copy.deepcopy(root)
    selected = []
    for selector in selectors:
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError('Nonempty CSS selector required')
        nodes = work.select(selector)
        if not nodes:
            raise ValueError('Reviewed selector has no matches')
        selected.extend(nodes)
    selected_ids = {id(n) for n in selected}
    def deleted(node):
        if node.name in ('del', 's', 'strike'):
            return True
        declarations = [v.split(':', 1) for v in node.get('style', '').split(';') if ':' in v]
        return any(key.strip().lower() in ('text-decoration', 'text-decoration-line')
                   and re.search(r'(?<![-\w])line-through(?![-\w])', value.lower())
                   for key, value in declarations)
    all_deleted = [n for n in work.find_all(True) if deleted(n)]
    if {id(n) for n in all_deleted} != selected_ids:
        raise ValueError('Selections must cover all and only explicit deletion nodes')
    metadata = []
    for node in all_deleted:
        if node.name not in ('span', 'del', 's', 'strike'):
            raise ValueError('Unsupported deletion container')
        if node.name == 'span':
            decorations = [v for v in node.get('style', '').split(';')
                           if ':' in v and v.split(':', 1)[0].strip().lower()
                           in ('text-decoration', 'text-decoration-line')]
            if len(decorations) != 1:
                raise ValueError('Conflicting decoration declarations')
        if node.find(True) is not None:
            raise ValueError('Nested deletion content requires separate review')
        value = ' '.join(node.get_text(' ', strip=True).split())
        if not value or ']' in value:
            raise ValueError('Empty or ambiguous deletion text')
        metadata.append(dict(tag=node.name, text=value, original_markup=str(node)))
        node.replace_with(NavigableString(f' {DELETION_PREFIX}{value}] '))
    return dict(text=' '.join(work.get_text(' ', strip=True).split()),
                deletions=metadata, marker_interpretation='Deleted in original source; no current-rule inference.')
