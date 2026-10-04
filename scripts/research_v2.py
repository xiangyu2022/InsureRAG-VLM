"""Independent v2 integrity/scoring/statistics helpers; v1 definitions stay intact."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import numpy as np

from scripts import eval_research_benchmark as v1

SCORE_VERSION = 'research_v2_context_contract_v1'
SUMMARY_VERSION = 'document_cluster_bootstrap_v1'
PROTOCOL_VERSION = 'prepared_document_benchmark_v2_1'
METRICS = (
    'gold_evidence_in_context',
    'supported_context_contract_success', 'supported_answer_coverage',
    'conditional_supported_contract_accuracy', 'answer_coverage',
    'conditional_answer_contract_precision', 'unsupported_strict_abstention',
    'unsupported_answer_rate', 'unsupported_failure_to_strictly_abstain', 'supported_strict_abstention',
    'retrieval_hit_at_k', 'generation_complete', 'json_contract_success', 'request_error_rate',
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest_object(value):
    return v1.sha(canonical(value))


def normalized_question(value):
    return re.sub(r'\W+', ' ', str(value).casefold()).strip()


def score_response(raw, item, pages_by_source, ranked_sources, *, prompt, generation, error=None):
    complete = (not error and generation.get('done') is True
                and generation.get('done_reason') == 'stop' and not generation.get('truncated', False))
    result = v1.score_response(raw, item, pages_by_source, ranked_sources,
        observed_context=v1.context_from_prompt(prompt), generation_complete=complete)
    result['base_contract_version'] = result['score_version']
    result['score_version'] = SCORE_VERSION
    parsed = result.get('parsed', {})
    result['answered'] = bool(complete and result['valid_json'] and not parsed.get('abstain', True)
                              and str(parsed.get('answer', '')).strip())
    result['request_error'] = bool(error)
    sections = v1.supplied_source_sections(v1.context_from_prompt(prompt))
    result['gold_evidence_in_context'] = bool(item['answerable'] and any(
        v1.whitespace(gold['evidence_span']) in v1.whitespace(section)
        for gold in item['gold'] for section in sections.get(gold['source'], [])
    )) if item['answerable'] else None
    return result


def metric_pair(row, metric):
    """Return (success, denominator contribution); errors never disappear."""
    scores = row['scores']
    supported = row['answerable'] is True
    answered = bool(scores.get('answered'))
    correct = bool(scores.get('context_grounded_key_pass'))
    strict_decline = bool(scores.get('completed_strict_abstention'))
    if metric == 'gold_evidence_in_context':
        eligible = supported and row['mode'] == 'retrieved'
        return int(eligible and bool(scores.get('gold_evidence_in_context'))), int(eligible)
    if metric == 'supported_context_contract_success':
        return int(correct and supported), int(supported)
    if metric == 'supported_answer_coverage':
        return int(answered and supported), int(supported)
    if metric == 'conditional_supported_contract_accuracy':
        return int(correct and supported and answered), int(supported and answered)
    if metric == 'answer_coverage':
        return int(answered), 1
    if metric == 'conditional_answer_contract_precision':
        return int(correct and supported and answered), int(answered)
    if metric == 'unsupported_strict_abstention':
        return int(strict_decline and not supported), int(not supported)
    if metric == 'unsupported_answer_rate':
        return int(answered and not supported), int(not supported)
    if metric == 'unsupported_failure_to_strictly_abstain':
        return int(not strict_decline and not supported), int(not supported)
    if metric == 'supported_strict_abstention':
        return int(strict_decline and supported), int(supported)
    if metric == 'retrieval_hit_at_k':
        eligible = supported and row['mode'] == 'retrieved'
        return int(eligible and bool(scores.get('retrieval_hit'))), int(eligible)
    if metric == 'generation_complete':
        return int(bool(scores.get('generation_complete'))), 1
    if metric == 'json_contract_success':
        return int(bool(scores.get('valid_json'))), 1
    if metric == 'request_error_rate':
        return int(bool(row.get('error'))), 1
    raise ValueError('Unknown metric: ' + metric)


def ratio_summary(rows, metric, *, bootstrap=2000, seed=42, cluster_key='document_id'):
    if bootstrap < 1:
        raise ValueError('Bootstrap draws must be positive')
    clusters = defaultdict(lambda: [0, 0])
    for row in rows:
        numerator, denominator = metric_pair(row, metric)
        pair = clusters[row[cluster_key]]
        pair[0] += numerator
        pair[1] += denominator
    counts = np.asarray([clusters[key] for key in sorted(clusters)], dtype=np.int64).reshape(-1, 2)
    numerator, denominator = counts.sum(axis=0) if len(counts) else (0, 0)
    eligible_clusters = int(np.count_nonzero(counts[:, 1])) if len(counts) else 0
    result = {'successes': int(numerator), 'total': int(denominator),
              'rate': float(numerator / denominator) if denominator else None,
              'cluster_unit': cluster_key, 'clusters': len(counts),
              'clusters_with_denominator': eligible_clusters,
              'ci95_low': None, 'ci95_high': None,
              'ci_method': 'percentile_cluster_bootstrap', 'bootstrap_draws': bootstrap,
              'bootstrap_seed': seed, 'valid_bootstrap_draws': 0, 'undefined_bootstrap_draws': 0}
    if eligible_clusters < 2:
        result['ci_unavailable_reason'] = 'fewer_than_two_clusters_with_metric_denominator'
        return result
    rng = np.random.default_rng(seed)
    sampled = counts[rng.integers(0, len(counts), size=(bootstrap, len(counts)))].sum(axis=1)
    valid = sampled[:, 1] > 0
    rates = sampled[valid, 0] / sampled[valid, 1]
    result['valid_bootstrap_draws'] = int(valid.sum())
    result['undefined_bootstrap_draws'] = int((~valid).sum())
    if len(rates):
        result['ci95_low'], result['ci95_high'] = [float(x) for x in np.quantile(rates, [0.025, 0.975])]
    else:
        result['ci_unavailable_reason'] = 'all_bootstrap_denominators_zero'
    if eligible_clusters < 10:
        result['small_cluster_warning'] = 'Fewer than ten eligible clusters; percentile intervals may be unstable.'
    return result


def metric_table(rows, *, bootstrap=2000, seed=42, cluster_key='document_id'):
    return {name: ratio_summary(rows, name, bootstrap=bootstrap, seed=seed, cluster_key=cluster_key)
            for name in METRICS}


def summarize(rows, *, bootstrap=2000, seed=42):
    if any(row.get('scores', {}).get('score_version') != SCORE_VERSION for row in rows):
        raise ValueError('Cannot mix v1 or incompatible score versions into v2')
    result = {'summary_version': SUMMARY_VERSION, 'score_version': SCORE_VERSION,
              'primary_metric': 'supported_context_contract_success',
              'primary_retrieval_metric': 'gold_evidence_in_context',
              'interpretation': 'Deterministic contract success, not semantic insurance accuracy; authored document clusters are not a representative population sample.',
              'coverage_definition': 'Completed, schema-valid, non-abstaining output with a nonempty answer.',
              'conditional_accuracy_definition': 'Strict context-contract successes among answered supported questions; report together with coverage.',
              'bootstrap': {'draws': bootstrap, 'seed': seed, 'unit': 'document_id',
                            'weighting': 'Question-weighted ratio after resampling whole documents with replacement.'},
              'modes': {}}
    for mode in sorted({row['mode'] for row in rows}):
        group = [row for row in rows if row['mode'] == mode]
        by_document, by_category = defaultdict(list), defaultdict(list)
        for row in group:
            by_document[row['document_id']].append(row)
            by_category[row['category']].append(row)
        result['modes'][mode] = {
            'cases': len(group), 'documents': len(by_document),
            'document_families': len({row['document_family_id'] for row in group}),
            'answerable_cases': sum(row['answerable'] for row in group),
            'unsupported_cases': sum(not row['answerable'] for row in group),
            'overall': metric_table(group, bootstrap=bootstrap, seed=seed),
            'family_cluster_sensitivity': metric_table(group, bootstrap=bootstrap, seed=seed, cluster_key='document_family_id'),
            'by_document': {key: metric_table(value, bootstrap=bootstrap, seed=seed) for key, value in sorted(by_document.items())},
            'by_category': {key: metric_table(value, bootstrap=bootstrap, seed=seed) for key, value in sorted(by_category.items())},
        }
    return result


def paired_metric(left_rows, right_rows, metric, *, bootstrap=2000, seed=42, cluster_key='document_id'):
    if bootstrap < 1:
        raise ValueError('Bootstrap draws must be positive')
    if len(left_rows) != len(right_rows):
        raise ValueError('Paired rows differ in length')
    clusters = defaultdict(lambda: [0, 0, 0, 0])
    for left, right in zip(left_rows, right_rows, strict=True):
        if (left['id'], left['mode'], left[cluster_key]) != (right['id'], right['mode'], right[cluster_key]):
            raise ValueError('Paired record/cluster identity differs')
        ln, ld = metric_pair(left, metric)
        rn, rd = metric_pair(right, metric)
        cell = clusters[left[cluster_key]]
        for index, value in enumerate((ln, ld, rn, rd)):
            cell[index] += value
    counts = np.asarray([clusters[key] for key in sorted(clusters)], dtype=np.int64).reshape(-1, 4)
    ln, ld, rn, rd = counts.sum(axis=0) if len(counts) else (0, 0, 0, 0)
    result = {'left_successes': int(ln), 'left_total': int(ld), 'right_successes': int(rn), 'right_total': int(rd),
              'right_minus_left_percentage_points': float(100 * (rn / rd - ln / ld)) if ld and rd else None,
              'ci95_low_pp': None, 'ci95_high_pp': None, 'cluster_unit': cluster_key,
              'clusters': len(counts), 'bootstrap_draws': bootstrap, 'bootstrap_seed': seed,
              'valid_bootstrap_draws': 0, 'undefined_bootstrap_draws': 0,
              'ci_method': 'paired_percentile_cluster_bootstrap'}
    if min(np.count_nonzero(counts[:, 1]), np.count_nonzero(counts[:, 3])) < 10:
        result['small_cluster_warning'] = 'Fewer than ten eligible clusters in an arm; percentile intervals may be unstable.'
    if np.count_nonzero(counts[:, 1]) < 2 or np.count_nonzero(counts[:, 3]) < 2:
        result['ci_unavailable_reason'] = 'fewer_than_two_eligible_clusters_in_an_arm'
        return result
    rng = np.random.default_rng(seed)
    sampled = counts[rng.integers(0, len(counts), size=(bootstrap, len(counts)))].sum(axis=1)
    valid = (sampled[:, 1] > 0) & (sampled[:, 3] > 0)
    differences = 100 * (sampled[valid, 2] / sampled[valid, 3] - sampled[valid, 0] / sampled[valid, 1])
    result['valid_bootstrap_draws'] = int(valid.sum())
    result['undefined_bootstrap_draws'] = int((~valid).sum())
    if len(differences):
        result['ci95_low_pp'], result['ci95_high_pp'] = [float(x) for x in np.quantile(differences, [0.025, 0.975])]
    return result


def render_summary(report):
    lines = ['# Expanded document-cluster diagnostic', '', report['interpretation'], '',
             '| Mode | Metric | Success / denominator | Rate | Document-cluster 95% interval |',
             '| --- | --- | ---: | ---: | --- |']
    for mode, group in report['modes'].items():
        for name, value in group['overall'].items():
            rate = 'n/a' if value['rate'] is None else f"{100 * value['rate']:.2f}%"
            interval = 'not estimable' if value['ci95_low'] is None else f"{100 * value['ci95_low']:.2f}%–{100 * value['ci95_high']:.2f}%"
            lines.append(f"| {mode} | {name} | {value['successes']}/{value['total']} | {rate} | {interval} |")
    lines += ['', 'Full per-document, per-category, and document-family sensitivity tables are in summary.json.',
              'Intervals resample whole documents and remain descriptive for this authored, non-random fixture. Conditional contract accuracy must be interpreted together with answer coverage and unsupported-question behavior.', '']
    return '\n'.join(lines)


def verify_fixture(folder, corpus, *, expected_counts=None):
    """Validate before retrieval; labels never determine the retrieval candidate set."""
    folder, corpus = Path(folder), Path(corpus)
    expected_counts = {'dev': 60, 'test': 240} if expected_counts is None else expected_counts
    lock = json.loads((folder / 'manifest.lock.json').read_text(encoding='utf-8'))
    required = {'dev.jsonl', 'test.jsonl', 'documents.json'}
    if not required <= set(lock.get('files', {})):
        raise ValueError('Fixture lock must include dev, test, and document registry')
    for name, expected in lock['files'].items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or v1.sha(path) != expected:
            raise ValueError('Frozen fixture file mismatch: ' + name)
    corpus_entries = lock.get('corpus_files', {})
    if {Path(name).name for name in corpus_entries} != {'rag_pages.jsonl', 'rag_snippets.jsonl'} or len(corpus_entries) != 2:
        raise ValueError('Fixture lock must identify both exact corpus files')
    for name, expected in corpus_entries.items():
        if v1.sha(corpus / Path(name).name) != expected:
            raise ValueError('Frozen corpus file mismatch: ' + name)
    for name, expected in lock.get('source_files', {}).items():
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or v1.sha(path) != expected:
            raise ValueError('Frozen original source file mismatch: ' + name)
    documents = json.loads((folder / 'documents.json').read_text(encoding='utf-8'))
    if not isinstance(documents, list):
        raise ValueError('documents.json must be a list')
    registry = {}
    families = defaultdict(set)
    for document in documents:
        for key in ('doc_id', 'document_family_id', 'title', 'split'):
            if not isinstance(document.get(key), str) or not document[key].strip():
                raise ValueError('Missing document field: ' + key)
        if document['split'] not in ('dev', 'test') or document['doc_id'] in registry:
            raise ValueError('Invalid split or duplicate document ID')
        registry[document['doc_id']] = document
        families[document['document_family_id']].add(document['split'])
        if document.get('source_kind') == 'official_pdf':
            source_name = 'sources/' + document['doc_id'] + '.pdf'
            if lock.get('source_files', {}).get(source_name) != document.get('pdf_sha256') or not document.get('pdf_sha256'):
                raise ValueError('Official PDF registry identity lacks its matching frozen original source')
    if any(len(splits) != 1 for splits in families.values()):
        raise ValueError('Document family crosses development/test split')
    items = []
    ids, questions, facts = set(), set(), set()
    for split in ('dev', 'test'):
        rows = v1.read_jsonl(folder / (split + '.jsonl'))
        if len(rows) != expected_counts[split] or lock.get('counts', {}).get(split) != len(rows):
            raise ValueError('Unexpected or unlocked ' + split + ' case count')
        for item in rows:
            for key in ('id', 'question', 'category', 'fact_id', 'document_family_id'):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError('Missing case field: ' + key)
            question = normalized_question(item['question'])
            if item['id'] in ids or question in questions or item['fact_id'] in facts:
                raise ValueError('Duplicate ID, normalized question, or fact_id')
            ids.add(item['id']); questions.add(question); facts.add(item['fact_id'])
            if item.get('split') != split or type(item.get('answerable')) is not bool:
                raise ValueError('Invalid case split or answerability flag')
            scope = item.get('document_scope')
            if not isinstance(scope, list) or len(scope) != 1 or scope[0] not in registry:
                raise ValueError('Every case requires exactly one registered document scope')
            document = registry[scope[0]]
            if document['split'] != split or document['document_family_id'] != item['document_family_id']:
                raise ValueError('Case and document registry split/family differ')
            items.append(item)
    pages = v1.read_jsonl(corpus / 'rag_pages.jsonl')
    snippets = v1.read_jsonl(corpus / 'rag_snippets.jsonl')
    by_source = {}
    page_duplicates = defaultdict(list)
    for page in pages:
        if page['citation'] in by_source:
            raise ValueError('Duplicate source citation in corpus')
        by_source[page['citation']] = page
        if page['doc_id'] in registry:
            normalized = v1.whitespace(page['text']).casefold()
            if len(normalized) >= 80:
                page_duplicates[v1.sha(normalized)].append(page)
    duplicate_groups = []
    for group in page_duplicates.values():
        if len(group) < 2:
            continue
        duplicate_groups.append([page['citation'] for page in group])
        if len({registry[page['doc_id']]['split'] for page in group}) > 1:
            raise ValueError('Duplicate normalized evidence page crosses development/test split')
    for snippet in snippets:
        if snippet['doc_id'] in registry and not any(page['doc_id'] == snippet['doc_id'] and page['page'] == snippet['page'] for page in pages):
            raise ValueError('Snippet has no corresponding source page')
    for item in items:
        if not any(page['doc_id'] == item['document_scope'][0] for page in pages):
            raise ValueError('Document scope has no corpus pages: ' + item['id'])
        gold = item.get('gold')
        if not isinstance(gold, list) or not isinstance(item.get('answer_keys'), list):
            raise ValueError('Case gold/answer_keys must be lists')
        if item['answerable']:
            if not gold or not v1.answer_key_match(item.get('reference_answer', ''), item['answer_keys']):
                raise ValueError('Supported reference lacks gold or fails frozen answer keys: ' + item['id'])
            for evidence in gold:
                page = by_source.get(evidence.get('source'))
                span = evidence.get('evidence_span')
                if (not page or page['doc_id'] not in item['document_scope'] or not isinstance(span, str)
                        or not span.strip() or v1.whitespace(span) not in v1.whitespace(page['text'])
                        or ('page' in evidence and evidence['page'] != page['page'])):
                    raise ValueError('Gold evidence is absent from the exact scoped source: ' + item['id'])
        elif gold or item['answer_keys']:
            raise ValueError('Unsupported case contains gold evidence/keys: ' + item['id'])
    audit = {'counts': dict(Counter(item['split'] for item in items)),
             'document_counts': dict(Counter(doc['split'] for doc in documents)),
             'family_counts': dict(Counter(next(iter(splits)) for splits in families.values())),
             'categories': dict(Counter(item['category'] for item in items)),
             'verified_original_source_files': len(lock.get('source_files', {})),
             'same_split_duplicate_page_groups': duplicate_groups,
             'duplicate_guards': ['unique case IDs', 'unique normalized questions', 'unique authored fact IDs',
                                  'document/family split separation', 'no exact normalized page duplicates across splits'],
             'limitation': 'Declared families and exact-page duplicate checks do not detect all paraphrased or partial near-duplicates, nor pretraining exposure.'}
    return lock, items, pages, snippets, audit


def source_hashes(root):
    paths = sorted((root / 'src/insurerag_vlm').glob('*.py'))
    paths += [root / 'scripts' / name for name in (
        'eval_research_benchmark.py', 'research_v2.py', 'prepare_research_benchmark_v2.py',
        'eval_research_benchmark_v2.py', 'compare_research_runs_v2.py')]
    return {path.relative_to(root).as_posix(): v1.sha(path) for path in paths}


def load_prepared(folder, *, root=None, check_code=True):
    folder = Path(folder)
    protocol = json.loads((folder / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('protocol_version') != PROTOCOL_VERSION or protocol.get('score_version') != SCORE_VERSION:
        raise ValueError('Unsupported prepared protocol or score version')
    if protocol.get('prompt_contract_sha256') != digest_object(protocol.get('prompt_contract')):
        raise ValueError('Prepared prompt contract checksum mismatch')
    if protocol['prompt_contract'] != {'system': v1.SYSTEM, 'template': v1.PROMPT_TEMPLATE, 'schema': v1.SCHEMA}:
        raise ValueError('Prepared prompt contract differs from the implemented transport')
    if set(protocol.get('prepared_files', {})) != {'inputs.jsonl', 'pages.jsonl'}:
        raise ValueError('Prepared protocol lacks exact input/page checksums')
    for name, expected in protocol['prepared_files'].items():
        if v1.sha(folder / name) != expected:
            raise ValueError('Frozen prepared input changed: ' + name)
    if check_code and (root is None or protocol.get('code_sha256') != source_hashes(Path(root))):
        raise ValueError('Evaluation code differs from the frozen protocol; create a new preparation/version')
    inputs = v1.read_jsonl(folder / 'inputs.jsonl')
    pages = v1.read_jsonl(folder / 'pages.jsonl')
    schedule = [{'id': row['id'], 'mode': row['mode']} for row in inputs]
    if schedule != protocol.get('request_schedule') or len(schedule) != len({(r['id'], r['mode']) for r in schedule}):
        raise ValueError('Prepared request schedule is incomplete or duplicated')
    for row in inputs:
        if row.get('prompt_sha256') != v1.sha(row['prompt']) or row.get('annotation_sha256') != digest_object(row['case']):
            raise ValueError('Prepared prompt or annotation checksum mismatch')
        if row['question'] != row['case']['question'] or row['id'] != row['case']['id']:
            raise ValueError('Prepared case identity differs from annotation')
    return protocol, inputs, pages
