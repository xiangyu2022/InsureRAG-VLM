"""Export a small review bundle without model outputs or source-corpus excerpts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def load(path): return json.loads(path.read_text(encoding='utf8'))
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def export(root):
    output = root / 'published'; output.mkdir(exist_ok=True)
    summary = copy.deepcopy(load(root / 'summary.json'))
    for arm in summary['arithmetic_arms'].values():
        for row in arm['details']:
            for citation in row.get('citations', []):
                quote = citation.pop('quote')
                citation['quote_utf8_sha256'] = hashlib.sha256(quote.encode('utf8')).hexdigest()
    write(output / 'summary.json', summary)
    for source, target in [('dev/protocol.json','protocol.json'), ('numeric_review.json','numeric_review.json'),
                           ('postprocessing_replay.json','postprocessing_replay.json'), ('verification.json','verification.json')]:
        write(output / target, load(root / source))
    cases = load(root / 'dev/cases.json'); retrieval = load(root / 'dev/retrieval.json')
    inventory = []
    for case in cases:
        r = retrieval[case['id']]
        inventory.append({k: case[k] for k in ('id','cohort','corpus','report','gold_answer_ids','annotation_origin','expected_abstain') if k in case} | {
            'question_utf8_sha256': hashlib.sha256(case['question'].encode('utf8')).hexdigest(),
            'top5_answer_ids': [a['answer_id'] for a in r['results']],
            'intervention_base_id': case.get('intervention_base_id'),
            'intervention_donor_id': r.get('intervention_donor_id')})
    write(output / 'case_inventory.json', inventory)
    # Include model configuration and timing provenance, never complete prompts.
    environments = {}
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and (folder / 'environment.json').exists():
            value = load(folder / 'environment.json')
            environments[folder.name] = {k: value[k] for k in ('model','started_utc','prompt_mode','json_input_encoding',
                'purpose','diagnostic_protocol_sha256','script_sha256','answer_integration_sha256','validator_sha256') if k in value}
    write(output / 'run_inventory.json', environments)
    # A hash inventory supports local audit and reconstruction without uploading
    # hundreds of model answers or redistributing third-party source corpora.
    manifest = {'raw_outputs_published': False, 'source_excerpts_published': False, 'model_weights_published': False,
                'local_evidence': {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in sorted(root.rglob('*')) if p.is_file() and output not in p.parents},
                'published_files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in sorted(output.iterdir()) if p.is_file() and p.name != 'artifact_manifest.json'}}
    write(output / 'artifact_manifest.json', manifest)
    print(json.dumps({'published_summary_files': len(list(output.iterdir())),
                      'bytes': sum(p.stat().st_size for p in output.iterdir()), 'destination': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    export(parser.parse_args().root)
