"""Actual full-parameter MiniLM fine-tuning on cleaned, author-labeled insurance questions."""
import argparse
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
from scripts.prepare_insuranceqa import read_jsonl, write_json, sha


def sample_group(group, rng, epoch):
    positive = group['positive_ids'][(epoch-1) % len(group['positive_ids'])]
    negatives = []
    for category, count in [('dense_hard', 2), ('lexical_hard', 2), ('random', 1)]:
        pool = [x for x in group['negative_ids'] if group['negative_sources'][x] == category]
        if len(pool) < count:
            raise ValueError('Training group lacks its declared negative budget')
        negatives.extend(rng.sample(pool, count))
    if positive in negatives or len(set(negatives)) != 5:
        raise ValueError('Invalid positive/negative training group')
    return [positive] + negatives


def run(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not torch.cuda.is_available():
        raise ValueError('This recorded training protocol requires a CUDA GPU')
    torch.set_num_threads(4); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.data/'manifest.lock.json').read_text(encoding='utf8'))
    for name, expected in manifest['files'].items():
        if sha(args.data/name) != expected: raise ValueError('Training fixture changed: '+name)
    groups = read_jsonl(args.data/'train_groups.jsonl')
    answers = {r['id']: r['text'] for r in read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl')}
    args.output.mkdir(parents=True, exist_ok=False); args.checkpoints.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True,
        trust_remote_code=False, dtype=torch.float32).to('cuda')
    if model.config.num_labels != 1: raise ValueError('Expected one relevance logit')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.01, fused=True)
    steps_per_epoch = math.ceil(math.ceil(len(groups)/args.micro_batch)/args.accumulation)
    total_steps = steps_per_epoch*args.epochs; warmup = max(1, round(total_steps*.1))
    def lr_factor(step):
        if step < warmup: return (step+1)/warmup
        return max(0., (total_steps-step)/max(1, total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    base_files = {p.name: sha(p) for p in sorted(args.model.iterdir()) if p.is_file() and p.suffix in {'.json','.safetensors','.txt'}}
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
        'training_questions': len(groups), 'epochs': args.epochs, 'optimizer': 'AdamW',
        'learning_rate': args.learning_rate, 'weight_decay': .01, 'warmup_steps': warmup,
        'optimizer_steps_planned': total_steps, 'micro_batch_groups': args.micro_batch,
        'gradient_accumulation': args.accumulation, 'group_size': 6,
        'loss': 'listwise cross entropy, one author positive + 2 BGE hard + 2 BM25 hard + 1 random negative',
        'label_smoothing': .05, 'positive_policy': 'Cycle original positive answer IDs by epoch; no generated positive labels',
        'max_length': 512, 'precision': 'float32 parameters; bfloat16 autocast',
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'base_model_files_sha256': base_files, 'training_manifest_sha256': sha(args.data/'manifest.lock.json'),
        'code_sha256': sha(Path(__file__)), 'test_used_for_training_or_selection': False,
        'selection': 'Score each epoch on all 2000 validation questions with fixed candidate lists. Select epoch and cross weight on validation only.'}
    write_json(protocol, args.output/'protocol.json')
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter(); steps = 0; total_pairs = 0; checkpoints = []
    optimizer.zero_grad(set_to_none=True)
    with (args.output/'training_log.jsonl').open('w', encoding='utf8') as log:
        for epoch in range(1, args.epochs+1):
            model.train(); rng = random.Random(args.seed+epoch); order = list(range(len(groups))); rng.shuffle(order)
            epoch_loss = 0.; epoch_groups = 0; micro_batches = math.ceil(len(order)/args.micro_batch)
            for batch_idx, offset in enumerate(range(0, len(order), args.micro_batch)):
                batch = [groups[i] for i in order[offset:offset+args.micro_batch]]
                qtexts = []; passages = []
                for g in batch:
                    ids = sample_group(g, rng, epoch)
                    qtexts.extend([g['question']]*len(ids)); passages.extend(answers[i] for i in ids)
                inputs = tokenizer(qtexts, passages, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(**inputs).logits.reshape(len(batch), 6).float()
                    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(batch), dtype=torch.long, device='cuda'), label_smoothing=.05)
                if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                window_start = (batch_idx//args.accumulation)*args.accumulation
                window_size = min(args.accumulation, micro_batches-window_start)
                (loss/window_size).backward()
                epoch_loss += float(loss.detach())*len(batch); epoch_groups += len(batch); total_pairs += len(qtexts)
                if (batch_idx+1) % args.accumulation == 0 or batch_idx+1 == micro_batches:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm): raise ValueError('Non-finite training gradient')
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); steps += 1
                    if steps % 100 == 0 or batch_idx+1 == micro_batches:
                        row = {'epoch': epoch, 'optimizer_step': steps, 'epoch_groups_seen': epoch_groups,
                            'mean_epoch_loss': epoch_loss/epoch_groups, 'learning_rate': optimizer.param_groups[0]['lr'],
                            'seconds': round(time.perf_counter()-start, 2), 'pairs_seen': total_pairs}
                        log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
            model.eval(); folder = args.checkpoints/f'epoch-{epoch}'; folder.mkdir()
            model.save_pretrained(folder, safe_serialization=True); tokenizer.save_pretrained(folder)
            checkpoint = {'epoch': epoch, 'optimizer_steps': steps, 'training_groups_seen': epoch_groups,
                'mean_loss': epoch_loss/epoch_groups, 'weights_sha256': sha(folder/'model.safetensors'),
                'fine_tuned_in_this_project': True, 'base_model': str(args.model),
                'training_manifest_sha256': protocol['training_manifest_sha256'],
                'training_protocol_sha256': sha(args.output/'protocol.json'), 'training_code_sha256': protocol['code_sha256']}
            write_json(checkpoint, folder/'training_provenance.json'); checkpoints.append(checkpoint)
            write_json({'completed_checkpoints': checkpoints, 'current_epoch': epoch}, args.output/'checkpoints.json')
    if sha(Path(__file__)) != protocol['code_sha256']: raise ValueError('Training code changed during run')
    write_json({'status': 'completed', 'optimizer_steps': steps, 'pairs_seen': total_pairs,
        'epochs': args.epochs, 'seconds': time.perf_counter()-start,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'base_weights_sha256': base_files['model.safetensors'], 'checkpoints': checkpoints,
        'weights_changed': all(c['weights_sha256'] != base_files['model.safetensors'] for c in checkpoints),
        'log_sha256': sha(args.output/'training_log.jsonl')}, args.output/'completion.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=3); p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--micro-batch', type=int, default=4); p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    run(p.parse_args())
