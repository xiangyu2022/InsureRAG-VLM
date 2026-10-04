#!/usr/bin/env python3
"""Remove the image from the four unique frozen visual prompts; diagnostic only."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_visual_counterfactual import SYSTEM, SCHEMA, verify_fixture, score_response
from scripts.eval_research_benchmark import read_jsonl, sha, write_json
from src.insurerag_vlm.vlm import VLMClient


def main(args):
    endpoint = urlparse(args.endpoint)
    if endpoint.scheme != 'http' or endpoint.hostname not in {'localhost', '127.0.0.1', '::1'} or endpoint.username or endpoint.password:
        raise ValueError('Explicit local HTTP endpoint required')
    lock, cases = verify_fixture(ROOT / 'data/benchmarks/visual_counterfactual_v1')
    previous = json.loads((args.image_run / 'run_metadata.json').read_text(encoding='utf-8'))
    originals = read_jsonl(args.image_run / 'predictions.jsonl')
    if previous['system_prompt'] != SYSTEM or previous['json_schema'] != SCHEMA:
        raise ValueError('Image-run prompt/schema differs')
    unique = {}
    for case in cases:
        unique.setdefault(case['field'], case['question'])
    assert len(unique) == 4
    for row in originals:
        assert row['request_text'] == {'system': SYSTEM, 'user': unique[row['field']]}
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True)
    client = VLMClient('ollama:' + previous['backend']['resolved_model'], provider='ollama',
        ollama_base_url=args.endpoint, generation_options=previous['generation_options'],
        thinking=False, request_timeout=300, use_hf_api=False, hf_api_token=None,
        openai_api_key=None, anthropic_api_key=None)
    if client.backend_metadata()['model_digest'] != previous['backend']['model_digest']:
        raise ValueError('Installed model differs from original image run')
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'four unique original prompts with images removed; no label or transcription supplied',
        'fixture_lock_sha256': sha(ROOT / 'data/benchmarks/visual_counterfactual_v1/manifest.lock.json'),
        'original_predictions_sha256': sha(args.image_run / 'predictions.jsonl'),
        'system_prompt': SYSTEM, 'json_schema': SCHEMA, 'questions': unique,
        'backend': client.backend_metadata(), 'generation_options': previous['generation_options'],
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / 'src/insurerag_vlm/vlm.py']}}
    write_json(metadata, args.output / 'run_metadata.json')
    rows = []
    for field, question in unique.items():
        raw = client.generate_chat(SYSTEM, question, response_format=SCHEMA)
        generation = client.last_generation_metadata
        assert generation['input_modality'] == 'text' and generation['image_count'] == 0
        scores = score_response({'answerable': False}, raw, generation)
        row = {'field': field, 'question': question, 'raw_response': raw,
               'generation': generation, 'scores': scores}
        rows.append(row)
        with (args.output / 'predictions.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        print(field, raw, flush=True)
    metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(metadata, args.output / 'run_metadata.json')
    write_json({'unique_prompts': len(rows),
        'strict_empty_refusals': sum(r['scores']['strict_abstention'] for r in rows),
        'limit': 'Mechanism control on four repeated fixture question forms, not a population accuracy or hallucination estimate. No image or OCR text was supplied.'},
        args.output / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-run', type=Path, default=ROOT / 'reports/visual_counterfactual_v1/qwen35_v1')
    parser.add_argument('--endpoint', default='http://127.0.0.1:11435')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
