"""Label-independent FinQA evidence serialization; no gold text enters passages."""
import hashlib
import re


def table_row_text(header, row):
    """Same serialization for all rows, following the corrected upstream template."""
    if len(header) != len(row): raise ValueError('Ragged table row')
    prefix = header[0]+' ' if header[0] else ''
    return ' '.join((prefix+''.join('the '+row[0]+' of '+head+' is '+cell+' ; '
                                   for head, cell in zip(header[1:], row[1:]))).split())


def page_evidence(record):
    """Uses only source fields, independent of question, labels and baseline outputs."""
    page = record['filename']
    rows = []
    for i, text in enumerate(record['pre_text']+record['post_text']):
        rows.append((f'text_{i}', text.strip(), 'text'))
    table = record['table']
    for i, row in enumerate(table):
        rows.append((f'table_{i}', table_row_text(table[0], row), 'table'))
    return [{'id': 'finqa_a_'+hashlib.sha256((page+'#'+key).encode()).hexdigest()[:24],
             'text': text, 'domain': 'financial_report', 'source_group': page.rsplit('/', 1)[0],
             'source_page': page, 'upstream_evidence_key': key, 'evidence_type': kind}
            for key, text, kind in rows]


def normalized_evidence(text):
    return re.sub(r'\s+', '', text).lower()
