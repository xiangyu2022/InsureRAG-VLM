"""Add six source-inspected page references; no QA labels are read."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import eval_research_benchmark as v1

# Inspected against the archived original PDF text; NJ printed numbering differs
# from physical numbering by two pages. This is a tiny annotated pilot, not an
# automatic or comprehensive graph extraction claim.
REFERENCES = [
    ('fema_nfip_claims_2024', 15, 17, '17', 'More on this can be found on page 17', 'REPAIRING & REBUILDING'),
    ('fema_nfip_claims_2024', 15, 9, '9', 'Return to page 9 for a list of what should be included in your records.', 'ORGANIZE LISTS & DOCUMENTATION'),
    ('fema_nfip_claims_2024', 15, 6, '6', 'Review coverage scenarios for examples of what the adjuster might consider on page 6.', 'Applying the Definition'),
    ('nj_auto_guide_2026', 4, 12, '10', 'See page 10 for lawsuit options', 'The Right to Sue'),
    ('nj_auto_guide_2026', 6, 7, '5', 'See chart on page 5 for differences between the STANDARD and BASIC polices.', 'The chart below compares the differences between the STANDARD and BASIC policies:'),
    ('wi_ltc_pi047', 28, 5, '5', 'CBRFs are covered only if your policy identifies these facilities as a covered benefit and the facility has been licensed as a CBRF by DHS. (page 5)', 'Community-Based Long-Term Care'),
]


def build(output):
    source = ROOT/'data/research_corpus/expanded_public_v1'
    fixture = ROOT/'data/benchmarks/research_v2'
    original = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    for name, digest in original['files'].items():
        if v1.sha(source/name) != digest:
            raise ValueError('Changed source corpus')
    lock = json.loads((fixture/'manifest.lock.json').read_text())
    pages = v1.read_jsonl(source/'rag_pages.jsonl')
    lookup = {(p['doc_id'], p['page']): p for p in pages}
    norm = lambda t: ' '.join(t.split())
    annotations = []
    for doc, from_page, to_page, label, quote, target_quote in REFERENCES:
        path = fixture/'sources'/f'{doc}.pdf'
        digest = v1.sha(path)
        if digest != lock['source_files'][f'sources/{doc}.pdf']:
            raise ValueError('Changed source PDF')
        with fitz.open(path) as pdf:
            actual_source = pdf[from_page-1].get_text('text', sort=True)
            actual_target = pdf[to_page-1].get_text('text', sort=True)
        printed = re.search(r'(\d+)\s*$', actual_target)
        if not printed or printed.group(1) != label:
            raise ValueError('Inspected target footer no longer matches printed page label')
        if quote not in norm(actual_source) or target_quote not in norm(actual_target):
            raise ValueError('Endpoint quote does not match original PDF')
        if quote not in lookup[(doc, from_page)]['text'] or target_quote not in lookup[(doc, to_page)]['text']:
            raise ValueError('Endpoint quote does not match curated text')
        ref = {'target_doc_id': doc, 'target_physical_page': to_page, 'target_printed_page_label': label,
               'evidence_span': quote, 'target_evidence_span': target_quote, 'source_pdf_sha256': digest,
               'annotation_author': 'AI source inspection; no domain-expert adjudication',
               'mapping_check': 'Native PDF target footer plus quoted target heading; NJ pages visually inspected'}
        lookup[(doc, from_page)].setdefault('source_references', []).append(ref)
        lookup[(doc, to_page)]['printed_page_label'] = label
        annotations.append({'source_doc_id': doc, 'source_physical_page': from_page, **ref})
    output.mkdir(parents=True, exist_ok=False)
    (output/'rag_pages.jsonl').write_text(''.join(v2canonical(p)+'\n' for p in pages), encoding='utf8')
    shutil.copyfile(source/'rag_snippets.jsonl', output/'rag_snippets.jsonl')
    (output/'reference_annotations.json').write_text(json.dumps(annotations, indent=2)+'\n', encoding='utf8')
    manifest = {**original, 'version': 'source_linked_public_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_corpus_manifest_sha256': v1.sha(source/'manifest.json'), 'source_inspected_reference_count': len(annotations),
                'reference_documents': 3, 'qa_labels_used_for_reference_construction': False,
                'status': 'Exploratory source-link pilot added after primary test evaluation; not a new held-out benchmark',
                'files': {n: v1.sha(output/n) for n in ['rag_pages.jsonl', 'rag_snippets.jsonl', 'reference_annotations.json']}}
    manifest['limitations'] += ['Six literal page references in three guides; does not establish a comprehensive insurance knowledge graph.',
        'Source reference presence is checked; correctness and legal applicability of the published reference are not established.']
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps({'references': len(annotations), 'documents': 3, 'corpus_counts': original['counts']}))


def v2canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'data/research_corpus/source_linked_public_v1')
    build(p.parse_args().output)
