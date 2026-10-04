"""Replay frozen model outputs through final packing/guards, without generation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.answer_evidence_reranker import answer_retrieval
from scripts.compare_qwen35_grounding import digest, write


class ReplayClient:
    def __init__(self, row): self.row = row
    def generate(self, prompt):
        assert prompt == self.row['prompt']
        return self.row['raw_answer']
    def generate_chat(self, system, prompt):
        assert system == self.row['system_prompt'] and prompt == self.row['prompt']
        return self.row['raw_answer']
    def backend_metadata(self): return self.row['generation']
    def answer_trace(self, **kwargs):
        return {'generation_used': True, 'answer_backend': self.row['served']['answer_backend'],
                'backend_metadata': self.row['generation']}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--arm-prefix', default='utf8_')
    args = p.parse_args(); root = args.root
    retrieval = json.loads((root / 'dev/retrieval.json').read_text(encoding='utf8'))
    results = {}
    for arm in ('baseline_4b', 'baseline_9b', 'revised_4b', 'revised_9b'):
        path = root / (args.arm_prefix + arm) / 'predictions.jsonl'
        completion = json.loads((root / (args.arm_prefix + arm) / 'completion.json').read_text(encoding='utf8'))
        assert digest(path) == completion['predictions_sha256']
        rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]
        changed = []
        for row in rows:
            replay = answer_retrieval(retrieval[row['id']], ReplayClient(row),
                                      prompt_mode='research' if arm.startswith('revised') else 'main')
            if replay['served'] != row['served']: changed.append(row['id'])
        results[arm] = {'count': len(rows), 'changed_served_outputs': changed, 'generation_calls': 0}
    output = {'hybrid_pipeline_sha256': digest(ROOT / 'src/insurerag_vlm/hybrid_pipeline.py'),
              'input_read_encoding': 'utf-8', 'arm_prefix': args.arm_prefix,
              'all_104_served_outputs_unchanged': all(not r['changed_served_outputs'] for r in results.values()), 'arms': results}
    write(root / 'postprocessing_replay.json', output)
    print(json.dumps(output))
    if not output['all_104_served_outputs_unchanged']: raise SystemExit(1)


if __name__ == '__main__': main()
