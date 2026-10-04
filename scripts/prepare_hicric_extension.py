"""Pinned HICRIC corpus extension and publisher-written Q/A transfer benchmark."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha

REVISION = 'e9304975feff9ccaaf15cf547f698590d00d78c3'
QUESTION = re.compile(r'(?m)^[ \t]*Q(?:uestion)?[ \t]*(\d{1,3})[ \t]*[.:)][ \t]*')
def clean(text): return ' '.join(text.split())
def digest(text): return hashlib.sha256(text.encode('utf8')).hexdigest()


def extract_pairs(text):
    starts = list(QUESTION.finditer(text)); accepted = []; rejected = []
    for i, match in enumerate(starts):
        end = starts[i+1].start() if i+1 < len(starts) else len(text)
        answer_tag = re.search(r'(?m)^[ \t]*A(?:nswer)?[ \t]*'+re.escape(match.group(1))+r'[ \t]*[.:)][ \t]*', text[match.end():end])
        if not answer_tag:
            rejected.append({'offset': match.start(), 'reason': 'missing_matching_answer_tag'}); continue
        astart = match.end()+answer_tag.end(); qend = match.end()+answer_tag.start()
        question = clean(text[match.end():qend]); answer = clean(text[astart:end])
        reason = None
        if not question.endswith('?') or not 5 <= len(question.split()) <= 100:
            reason = 'not_a_standalone_question_or_question_length'
        elif not 25 <= len(answer.split()) <= 350:
            reason = 'answer_length_outside_25_350_words'
        elif re.search(r'\b(?:previous question|question\s+\d+|Q\d+|described above|discussed above|listed above)\b', question, re.I):
            reason = 'question_depends_on_omitted_context'
        if reason:
            rejected.append({'offset': match.start(), 'reason': reason}); continue
        accepted.append({'question': question, 'answer': answer, 'question_number': match.group(1),
                         'question_span': [match.end(), qend], 'answer_span': [astart, end]})
    return accepted, rejected


def run(args):
    from huggingface_hub import snapshot_download
    import pyarrow.parquet as pq
    download = args.cache
    snapshot_download('Persius/hicric', repo_type='dataset', revision=REVISION, local_dir=download,
        allow_patterns=['README.md','contract-coverage-rule-medical-policy/*.parquet','regulatory-guidance/*.parquet'])
    args.corpus.mkdir(parents=True, exist_ok=False); args.benchmark.mkdir(parents=True, exist_ok=False)
    records = {}; raw_counts = Counter(); raw_files = {}
    for path in sorted(download.rglob('*.parquet')):
        raw_files[path.relative_to(download).as_posix()] = sha(path)
        for row_index, row in enumerate(pq.read_table(path).to_pylist()):
            raw_counts[path.parent.name] += 1
            normalized = clean(row['text']); key = digest(normalized)
            provenance = {k: row[k] for k in ['source_url','source_md5','date_accessed','relative_path','tags']}
            provenance.update({'upstream_file': path.relative_to(download).as_posix(), 'upstream_row': row_index})
            if key in records:
                records[key]['provenance'].append(provenance); continue
            records[key] = {'id': 'hicric_'+key[:16], 'text': row['text'], 'content_sha256': key,
                            'category': path.parent.name, 'provenance': [provenance]}
    pairs = []; rejection = []; pages = []; snippets = []
    for rec in records.values():
        provenance = rec['provenance'][0]
        base = {'doc_id': rec['id'], 'page': 1, 'source_file': rec['id']+'.txt',
            'source_url': provenance['source_url'], 'citation': provenance['source_url'],
            'authority': 'Archived US government guidance, redistributed by HICRIC',
            'content_type': 'archived_coverage_rule' if rec['category'].startswith('contract') else 'archived_regulatory_guidance',
            'insurance_domain': ['health','medicare','medicaid'],
            'page_unit': 'Logical source record; not a physical PDF page',
            'dataset_revision': REVISION, 'record_content_sha256': rec['content_sha256']}
        pages.append({**base, 'record_id': 'rag_page::'+rec['id']+'::p0001', 'record_type': 'page', 'text': clean(rec['text'])})
        words = rec['text'].split()
        for i, offset in enumerate(range(0,len(words),120)):
            part = words[offset:offset+180]
            if not part: continue
            snippets.append({**base, 'record_id': 'rag_snippet::'+rec['id']+f'::p0001::s{i:04d}',
                             'record_type': 'snippet', 'text': ' '.join(part), 'start_word': offset, 'end_word': offset+len(part)})
            if offset+180 >= len(words): break
        if rec['category'] == 'regulatory-guidance':
            found, excluded = extract_pairs(rec['text'])
            pairs.extend({**p, 'record_id': rec['id'], 'source_url': provenance['source_url']} for p in found)
            rejection.extend({**r, 'record_id': rec['id']} for r in excluded)
    byquestion = defaultdict(list)
    for p in pairs: byquestion[clean(p['question']).lower()].append(p)
    cases = []; answer_records = {}; omitted = []
    for key, group in byquestion.items():
        if len({digest(p['answer']) for p in group}) != 1:
            omitted.append({'question': group[0]['question'], 'reason': 'same_question_different_historical_answers', 'n': len(group)}); continue
        p = group[0]; aid = 'gov_answer_'+digest(p['answer'])[:16]
        answer_records.setdefault(aid, {'id': aid, 'text': p['answer'], 'source_records': []})
        answer_records[aid]['source_records'].extend({k:g[k] for k in ['record_id','source_url','answer_span']} for g in group)
        cases.append({'id': 'gov_question_'+digest(key)[:16], 'question': p['question'],
            'gold_answer_ids': [aid], 'source_record_id': p['record_id'], 'source_url': p['source_url'],
            'question_span': p['question_span'], 'answer_span': p['answer_span'],
            'split': 'external_transfer_test', 'annotation_origin': 'Publisher-written numbered Q/A; deterministic source-bound extraction'})
    # Unseen benchmark: all cases are evaluation-only, no split is used for training.
    for name, rows, folder in [('records.jsonl',list(records.values()),args.corpus),('rag_pages.jsonl',pages,args.corpus),
        ('rag_snippets.jsonl',snippets,args.corpus),('questions.jsonl',cases,args.benchmark),('answers.jsonl',list(answer_records.values()),args.benchmark)]:
        with (folder/name).open('w',encoding='utf8') as handle:
            for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json({'rejected_sections': rejection, 'ambiguous_question_groups': omitted}, args.benchmark/'extraction_audit.json')
    shutil.copyfile(download/'README.md', args.corpus/'UPSTREAM_README.md')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'upstream': 'https://huggingface.co/datasets/Persius/hicric',
        'revision': REVISION, 'attribution': 'HICRIC Data, https://github.com/TPAFS/hicric',
        'curated_data_license': 'CC-BY-SA-4.0; upstream government sources retain their terms',
        'raw_record_counts': dict(raw_counts), 'unique_normalized_text_records': len(records),
        'duplicate_records_merged': sum(raw_counts.values())-len(records), 'snippets': len(snippets),
        'unique_source_urls': len({p['source_url'] for r in records.values() for p in r['provenance']}),
        'raw_files_sha256': raw_files, 'extraction_code_sha256': sha(Path(__file__)),
        'use': 'Separate archived corpus; not merged into the frozen 73-document regression corpus',
        'limitations': ['Record counts are not counts of independent PDF documents.', 'Historical guidance is not necessarily current law or coverage.'],
        'files': {n:sha(args.corpus/n) for n in ['records.jsonl','rag_pages.jsonl','rag_snippets.jsonl','UPSTREAM_README.md']}}
    write_json(manifest, args.corpus/'manifest.json')
    benchmark = {'created_utc': datetime.now(timezone.utc).isoformat(), 'questions': len(cases), 'answer_candidates': len(answer_records),
        'source_records': len({c['source_record_id'] for c in cases}), 'source_urls': len({c['source_url'] for c in cases}),
        'frozen_before_model_inference': True, 'training_examples': 0, 'hyperparameter_selection_examples': 0,
        'candidate_text_excludes_question': True, 'labels': 'Original publisher Q/A association, extracted; not expert adjudication of alternate relevant answers',
        'corpus_manifest_sha256': sha(args.corpus/'manifest.json'), 'extraction_code_sha256': sha(Path(__file__)),
        'limitations': ['Mechanical source-bound extraction; inspect the included extraction audit.',
          'Government FAQ answer retrieval is a transfer diagnostic, not full legal/policy reasoning.',
          'Questions from the same source are correlated; report source-cluster uncertainty.'],
        'files': {n:sha(args.benchmark/n) for n in ['questions.jsonl','answers.jsonl','extraction_audit.json']}}
    write_json(benchmark,args.benchmark/'manifest.lock.json')
    print(json.dumps({'corpus':{k:v for k,v in manifest.items() if k not in ['files','raw_files_sha256']},'benchmark':benchmark},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True);p.add_argument('--benchmark',type=Path,required=True)
    run(p.parse_args())
