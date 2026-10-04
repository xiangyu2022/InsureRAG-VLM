"""Full-corpus multi-positive query-encoder learning with original-BGE retention."""
import argparse
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl, sha, write_json
from scripts.prepare_reranker_training import normalize
from src.insurerag_vlm.query_adaptation import QUERY_INSTRUCTION, multi_positive_losses


def main(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    threadpool_limits(4)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available(): raise ValueError('CUDA is required for this registered run')
    run = ROOT / 'reports/query_adaptation_v1'
    planpath = run / 'selection_protocol.json'
    plan = json.loads(planpath.read_text(encoding='utf8'))
    assert args.seed in plan['seeds']
    for name, digest in plan['frozen_code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    data = ROOT / 'data/training/query_adaptation_v1'
    lock = json.loads((data / 'manifest.lock.json').read_text(encoding='utf8'))
    assert sha(data / 'manifest.lock.json') == plan['training_manifest_sha256']
    for name, digest in lock['files'].items(): assert sha(data / name) == digest
    cache = run / 'index_cache'
    cm = json.loads((cache / 'manifest.json').read_text(encoding='utf8'))
    for name in ['answer_embeddings', 'training_query_embeddings']:
        assert sha(cache / (name + '.npy')) == cm[name + '_sha256']
    groups = read_jsonl(data / 'train_groups.jsonl')
    ablation = json.loads((run / 'data_ablation_implementation.lock.json').read_text(encoding='utf8'))
    assert sha(Path(__file__)) == ablation['training_code_sha256']
    assert sha(run / 'data_ablation_protocol.json') == ablation['ablation_protocol_sha256']
    old_unique = [g for g in groups if g['source_domain'] != 'finance']
    old_slots = [i for i, g in enumerate(groups) if g['source_domain'] != 'finance'
                 for _ in range(plan['training']['repeat_factors'][g['source_domain']])]
    anchor_indices = list(range(len(groups)))
    replacements = []
    for i, g in enumerate(groups):
        if g['source_domain'] != 'finance': continue
        j = old_slots[int(hashlib.sha256(g['id'].encode()).hexdigest(), 16) % len(old_slots)]
        old = groups[j]
        anchor_indices[i] = j
        replacements.append({'optimization_slot_id': g['id'], 'replacement_question_id': old['id'],
                             'replacement_source_domain': old['source_domain']})
        groups[i] = {**g, 'question': old['question'], 'positive_ids': old['positive_ids'],
                     'replacement_question_id': old['id'], 'actual_source_domain': old['source_domain']}
    assert len(replacements) == 5464
    assert len({g['question'] for g in groups}) == len(old_unique)
    # source_domain remains the optimization slot: repeat and KL weights are held fixed.

    answers = read_jsonl(data / 'answers.jsonl')
    forbidden = set(json.loads((data / 'isolation.json').read_text(encoding='utf8'))['forbidden_answer_ids'])
    allowed = [i for i, a in enumerate(answers) if a['id'] not in forbidden]
    positions = {answers[i]['id']: j for j, i in enumerate(allowed)}
    aliases = defaultdict(list)
    for j, i in enumerate(allowed): aliases[normalize(answers[i]['text'])].append(j)
    lookup = {a['id']: a for a in answers}
    positive_columns = [sorted({j for a in g['positive_ids'] for j in aliases[normalize(lookup[a]['text'])]}) for g in groups]
    assert all(positive_columns)
    assert all(a in positions for g in groups for a in g['positive_ids'])
    docs = torch.tensor(np.load(cache / 'answer_embeddings.npy')[allowed], device='cuda')
    anchors = torch.tensor(np.load(cache / 'training_query_embeddings.npy')[anchor_indices], device='cuda')
    modelpath = ROOT / '../models/bge-small-en-v1.5'
    assert sha(modelpath / 'model.safetensors') == plan['initial_query_weights_sha256']
    tokenizer = AutoTokenizer.from_pretrained(modelpath, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(modelpath, local_files_only=True, trust_remote_code=False,
                                      dtype=torch.float32).to('cuda')
    out = run / f'ablation_train_seed_{args.seed}'
    models = ROOT / f'../models/insurerag-query-ablation-v1-seed-{args.seed}'
    out.mkdir(exist_ok=False)
    models.mkdir(exist_ok=False)
    write_json(replacements, out / 'replacement_mapping.json')
    recipe = plan['training']
    indices = [i for i, g in enumerate(groups) for _ in range(recipe['repeat_factors'][g['source_domain']])]
    micro, accumulation = recipe['micro_batch'], recipe['accumulation']
    per_epoch = math.ceil(math.ceil(len(indices)/micro)/accumulation)
    steps = per_epoch * recipe['epochs']
    warmup = round(.1*steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'], weight_decay=.01, fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s+1)/max(1,warmup) if s < warmup else max(0., (steps-s)/max(1,steps-warmup)))
    lengths = tokenizer([QUERY_INSTRUCTION + g['question'] for g in groups], truncation=False, verbose=False)['input_ids']
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
                'unique_training_questions': len(old_unique), 'optimization_question_slots': len(groups),
                'domains': dict(Counter(g['source_domain'] for g in old_unique)),
                'optimization_slot_domains': lock['source_domains'],
                'ablation_protocol_sha256': sha(run / 'data_ablation_protocol.json'),
                'ablation_training_code_sha256': sha(Path(__file__)),
                'additional_labelled_finance_questions': 0,
                'query_presentations_per_epoch': len(indices), 'eligible_document_vectors': len(allowed),
                'supervised_positive_pairs': sum(len(g['positive_ids']) for g in groups),
                'positive_text_aliases_are_also_positives': True,
                'full_candidate_denominator_not_sampled_negatives': True,
                'training': recipe, 'optimizer_steps_planned': steps,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'query_lengths_over_512': sum(len(t) > 512 for t in lengths),
                'selection_protocol_sha256': sha(planpath), 'training_manifest_sha256': sha(data / 'manifest.lock.json'),
                'document_index_sha256': sha(cache / 'answer_embeddings.npy'),
                'initial_weights_sha256': plan['initial_query_weights_sha256'],
                'code_sha256': plan['frozen_code_sha256'], 'test_used': False}
    write_json(protocol, out / 'protocol.json')
    rng = random.Random(args.seed)
    step = 0
    presented = 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    completions = []
    with (out / 'training_log.jsonl').open('w', encoding='utf8') as log:
        for epoch in range(1, recipe['epochs']+1):
            rng.shuffle(indices)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batches = math.ceil(len(indices)/micro)
            total_ce = total_kl = 0.
            n = 0
            for bi, offset in enumerate(range(0, len(indices), micro)):
                ix = indices[offset:offset+micro]
                batch = [groups[i] for i in ix]
                features = tokenizer([QUERY_INSTRUCTION+g['question'] for g in batch], padding=True,
                                      truncation=True, max_length=512, return_tensors='pt').to('cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    hidden = model(**features).last_hidden_state[:, 0]
                query = torch.nn.functional.normalize(hidden.float(), dim=-1)
                logits = query @ docs.T / recipe['temperature']
                with torch.no_grad(): teacher = anchors[ix] @ docs.T / recipe['temperature']
                mask = torch.zeros_like(logits, dtype=torch.bool)
                for j, i in enumerate(ix): mask[j, positive_columns[i]] = True
                ce, kl = multi_positive_losses(logits, mask, teacher)
                weight = torch.tensor([recipe['retention_weights'][g['source_domain']] for g in batch], device='cuda')
                loss = (ce + weight*kl).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                window_examples = min(micro*accumulation, len(indices)-(bi//accumulation)*micro*accumulation)
                (loss*len(ix)/window_examples).backward()
                n += len(ix)
                presented += len(ix)
                total_ce += float(ce.detach().sum())
                total_kl += float(kl.detach().sum())
                if (bi+1) % accumulation == 0 or bi+1 == batches:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step % 100 == 0 or bi+1 == batches:
                        row = {'seed': args.seed, 'epoch': epoch, 'step': step, 'mean_ce': total_ce/n,
                               'mean_kl': total_kl/n, 'query_presentations': presented,
                               'seconds': round(time.perf_counter()-start, 1)}
                        log.write(json.dumps(row)+'\n')
                        log.flush()
                        print(json.dumps(row), flush=True)
            folder = models / f'epoch-{epoch}'
            model.eval()
            model.save_pretrained(folder, safe_serialization=True)
            tokenizer.save_pretrained(folder)
            provenance = {'role': 'query_encoder_only', 'document_encoder': 'unchanged BAAI/bge-small-en-v1.5',
                          'query_instruction': QUERY_INSTRUCTION, 'pooling': 'cls', 'normalize': True, 'max_length': 512,
                          'seed': args.seed, 'epoch': epoch, 'optimizer_steps': step, 'query_presentations': presented,
                          'weights_sha256': sha(folder / 'model.safetensors'),
                          'document_encoder_weights_sha256': plan['initial_query_weights_sha256'],
                          'training_protocol_sha256': sha(out / 'protocol.json')}
            write_json(provenance, folder / 'query_adapter_config.json')
            completions.append(provenance)
    for name, digest in plan['frozen_code_sha256'].items(): assert sha(ROOT / name) == digest
    write_json({'status': 'completed', 'seconds': time.perf_counter()-start, 'optimizer_steps': step,
                'query_presentations': presented, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'epochs': completions, 'training_log_sha256': sha(out / 'training_log.jsonl')}, out / 'completion.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    main(parser.parse_args())
