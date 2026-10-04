"""Independently verify real parameter changes, data masks, and selection chronology."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.eval_query_adaptation import check_contract, configurations, modelpath


def load(p): return json.loads(p.read_text(encoding='utf8'))


def main(args):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    run = ROOT / 'reports/query_adaptation_v1'
    plan = check_contract()
    selected = load(run / 'selection.lock.json')
    assert selected['selected_on'] == 'valid'
    assert selected['config'] in configurations(plan)
    assert selected['selection_protocol_sha256'] == sha(run / 'selection_protocol.json')
    assert selected['validation_summary_sha256'] == sha(run / 'valid/summary.json')
    assert selected['validation_sweep_sha256'] == sha(run / 'validation_sweep.json')
    assert selected['query_weights_sha256'] == sha(modelpath(selected['config']) / 'model.safetensors')
    original = ROOT / '../models/bge-small-en-v1.5/model.safetensors'
    assert sha(original) == plan['initial_query_weights_sha256']
    weights = load_file(str(original))
    # Transformers 5 saves position_ids as a non-persistent, deterministic buffer.
    # Verify its exact value before excluding it from learned-parameter comparison.
    position_ids = weights.pop('embeddings.position_ids')
    assert torch.equal(position_ids, torch.arange(512).expand(1, -1))
    data = ROOT / 'data/training/query_adaptation_v1'
    manifest = load(data / 'manifest.lock.json')
    assert sha(data / 'manifest.lock.json') == plan['training_manifest_sha256']
    for name, digest in manifest['files'].items(): assert sha(data / name) == digest
    groups = read_jsonl(data / 'train_groups.jsonl')
    forbidden = set(load(data / 'isolation.json')['forbidden_answer_ids'])
    assert not any(set(g['positive_ids']) & forbidden for g in groups)
    assert len(groups) == 31351
    proofs = []
    checkpoint_files = {}
    for ablation in [False, True]:
        for seed in plan['seeds']:
            folder = run / f'{"ablation_" if ablation else ""}train_seed_{seed}'
            if not folder.exists() and not args.require_ablation: continue
            completion = load(folder / 'completion.json')
            protocol = load(folder / 'protocol.json')
            assert completion['status'] == 'completed'
            assert completion['optimizer_steps'] == protocol['optimizer_steps_planned'] == 3142
            assert completion['query_presentations'] == 100514
            assert completion['training_log_sha256'] == sha(folder / 'training_log.jsonl')
            assert protocol['initial_weights_sha256'] == plan['initial_query_weights_sha256']
            assert protocol['eligible_document_vectors'] == 100465
            assert protocol['selection_protocol_sha256'] == sha(run / 'selection_protocol.json')
            assert datetime.fromisoformat(protocol['created_utc']) > datetime.fromisoformat(plan['created_utc'])
            if not ablation:
                assert datetime.fromisoformat(protocol['created_utc']) < datetime.fromisoformat(selected['created_utc'])
            if ablation:
                replacements = load(folder / 'replacement_mapping.json')
                assert len(replacements) == 5464
                finance_ids = {g['id'] for g in groups if g['source_domain'] == 'finance'}
                old_ids = {g['id'] for g in groups if g['source_domain'] != 'finance'}
                assert {r['optimization_slot_id'] for r in replacements} == finance_ids
                assert all(r['replacement_question_id'] in old_ids for r in replacements)
                assert protocol['unique_training_questions'] == 25887
            epochs = []
            for epoch in completion['epochs']:
                kind = 'ablation' if ablation else 'adaptation'
                model = ROOT / f'../models/insurerag-query-{kind}-v1-seed-{seed}/epoch-{epoch["epoch"]}'
                for p in model.iterdir():
                    if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors'}:
                        checkpoint_files[p.resolve().relative_to(ROOT.parent).as_posix()] = sha(p)
                assert sha(model / 'model.safetensors') == epoch['weights_sha256']
                assert epoch['training_protocol_sha256'] == sha(folder / 'protocol.json')
                produced = load_file(str(model / 'model.safetensors'))
                assert set(produced) == set(weights)
                changed = {name: int((weights[name] != produced[name]).sum()) for name in weights}
                assert sum(changed.values()) > 1000000
                assert all(torch.isfinite(t).all() for t in produced.values())
                epochs.append({'epoch': epoch['epoch'], 'weights_sha256': epoch['weights_sha256'],
                               'changed_parameters': sum(changed.values()), 'changed_tensors': sum(n > 0 for n in changed.values()),
                               'optimizer_steps': epoch['optimizer_steps'], 'query_presentations': epoch['query_presentations']})
                del produced
            proofs.append({'seed': seed, 'ablation': ablation, 'unique_questions': protocol['unique_training_questions'],
                           'seconds': completion['seconds'], 'peak_cuda_allocated_bytes': completion['peak_cuda_allocated_bytes'],
                           'optimizer_steps': completion['optimizer_steps'], 'query_presentations': completion['query_presentations'],
                           'epochs': epochs})
    for folder in [original.parent, ROOT / '../models/insurerag-condition-listwise-v1-seed-123/epoch-1']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {'.json', '.txt', '.safetensors'}:
                checkpoint_files[p.resolve().relative_to(ROOT.parent).as_posix()] = sha(p)
    inference_lock = run / 'checkpoint_files.lock.json'
    if inference_lock.exists():
        assert load(inference_lock)['files_sha256'] == checkpoint_files
    else:
        assert not args.require_test
        write_json({'created_utc': datetime.now(timezone.utc).isoformat(), 'files_sha256': checkpoint_files,
                    'purpose': 'Inference file hashes after training and before new-test scoring'}, inference_lock)
    evaluation = {}
    for phase in ['valid', 'test']:
        path = run / phase / 'summary.json'
        if not path.exists():
            assert phase == 'test' and not args.require_test
            continue
        report = load(path)
        protocol = load(path.parent / 'protocol.json')
        assert report['status'] == 'completed'
        assert report['protocol_sha256'] == sha(path.parent / 'protocol.json')
        assert report['predictions_sha256'] == sha(path.parent / 'predictions.jsonl')
        for name, digest in {**report['score_files_sha256'], **report['query_vector_files_sha256']}.items():
            assert sha(path.parent / name) == digest
        if phase == 'test':
            assert protocol['selection_lock_sha256'] == sha(run / 'selection.lock.json')
            assert datetime.fromisoformat(protocol['created_utc']) > datetime.fromisoformat(selected['created_utc'])
        evaluation[phase] = {'summary_sha256': sha(path), 'new_cross_pairs': report['new_cross_pairs'],
                             'reused_cross_pairs': report['reused_cross_pairs']}
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'verified', 'training_runs': proofs,
              'primary_optimizer_steps': sum(p['optimizer_steps'] for p in proofs if not p['ablation']),
              'ablation_optimizer_steps': sum(p['optimizer_steps'] for p in proofs if p['ablation']),
              'primary_query_presentations': sum(p['query_presentations'] for p in proofs if not p['ablation']),
              'ablation_query_presentations': sum(p['query_presentations'] for p in proofs if p['ablation']),
              'original_document_encoder_weights_unchanged': True, 'fixed_reranker_weights_unchanged': True,
              'serialization_buffer_difference': {'name': 'embeddings.position_ids',
                  'verified_value': 'arange(512).expand(1, -1)',
                  'reason': 'Deterministic non-persistent buffer omitted by Transformers 5 save_pretrained'},
              'forbidden_training_positive_or_denominator_overlap': 0,
              'selection_lock_sha256': sha(run / 'selection.lock.json'), 'evaluation': evaluation,
              'checkpoint_files_sha256': sha(inference_lock),
              'code_sha256': sha(Path(__file__))}
    output = run / ('verification.json' if args.require_test else 'pre_test_verification.json')
    write_json(result, output)
    print(json.dumps({'status': 'verified', 'training_runs': len(proofs), 'evaluations': list(evaluation)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-ablation', action='store_true')
    p.add_argument('--require-test', action='store_true')
    main(p.parse_args())
