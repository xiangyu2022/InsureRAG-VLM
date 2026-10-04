"""Exercise all three actual-model research query routes without quality claims."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import sha, write_json

if __name__ == '__main__':
    results = []
    out = ROOT / 'reports/query_adaptation_v1/smoke'
    out.mkdir(exist_ok=False)
    for corpus, question in [
        ('insuranceqa', 'What does renters liability insurance cover?'),
        ('condition', 'When can a health plan impose cost sharing on preventive services?'),
        ('fiqa', 'How are term life and whole life insurance different?'),
    ]:
        target = out / f'{corpus}.json'
        command = [sys.executable, 'scripts/query_adapted_retrieval.py', '--corpus', corpus,
                   '--question', question, '--device', 'cuda', '--output', str(target)]
        with (out / f'{corpus}.log').open('w', encoding='utf8') as f:
            subprocess.run(command, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
        r = json.loads(target.read_text(encoding='utf8'))
        assert len(r['results']) == 3 and 1 <= r['candidate_count'] <= 200
        assert r['generation_calls'] == 0 and r['query_encoder_passes'] == 2
        results.append({'corpus': corpus, 'output_sha256': sha(target),
                        'seconds_including_loading': r['seconds_including_loading'],
                        'candidate_count': r['candidate_count'], 'top_answer_ids': [x['answer_id'] for x in r['results']]})
    write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                'quality_expert_adjudicated': False, 'runs': results,
                'query_script_sha256': sha(ROOT / 'scripts/query_adapted_retrieval.py')}, out / 'summary.json')
    print(json.dumps(results))
