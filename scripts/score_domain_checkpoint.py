"""Score an actual trained checkpoint over exactly the frozen baseline candidates."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from src.insurerag_vlm.domain_reranker import DomainCrossEncoder


def run(args):
    import torch
    torch.set_num_threads(4)
    candidates=read_jsonl(args.candidates/'scores.jsonl')
    original=json.loads((args.candidates/'protocol.json').read_text(encoding='utf8'))
    completed=json.loads((args.candidates/'completion.json').read_text(encoding='utf8'))
    if sha(args.candidates/'scores.jsonl')!=completed['scores_sha256']:raise ValueError('Candidate artifact changed')
    if args.split!=original['split']:raise ValueError('Split mismatch')
    selection=json.loads(args.selection.read_text(encoding='utf8')) if args.selection else None
    if args.split=='test':
        if selection is None or selection['selected_on']!='valid':raise ValueError('Test scoring needs frozen validation selection')
        if sha(args.model/'model.safetensors')!=selection['selected_weights_sha256']:raise ValueError('Wrong checkpoint for frozen selection')
    cases={r['id']:r['question'] for r in read_jsonl(ROOT/f'data/benchmarks/insuranceqa_v2/{args.split}.jsonl')}
    answers={r['id']:r['text'] for r in read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl')}
    args.output.mkdir(parents=True,exist_ok=False)
    model=DomainCrossEncoder(args.model,'cuda',64,512)
    files=[Path(__file__),ROOT/'src/insurerag_vlm/domain_reranker.py',ROOT/'src/insurerag_vlm/reranker.py']
    code={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'split':args.split,
        'candidate_source_sha256':sha(args.candidates/'scores.jsonl'),
        'candidate_source_protocol_sha256':sha(args.candidates/'protocol.json'),
        'selection_lock_sha256':sha(args.selection) if args.selection else None,
        'model':model.fingerprint(),'code_sha256':code,'questions':len(candidates),
        'inference_inputs':'Question and candidate answer text only; exact preexisting candidate IDs',
        'gold_label_injection':False}
    write_json(protocol,args.output/'protocol.json');start=time.perf_counter()
    with (args.output/'scores.jsonl').open('w',encoding='utf8') as handle:
        for start_idx in range(0,len(candidates),8):
            group=candidates[start_idx:start_idx+8]
            pairs=[(cases[r['id']],answers[a]) for r in group for a in r['candidate_ids']]
            scores=model.score_pairs(pairs);offset=0
            for row in group:
                n=len(row['candidate_ids']);updated={**row,'cross':scores[offset:offset+n].tolist()};offset+=n
                handle.write(json.dumps(updated)+'\n')
            handle.flush()
            done=start_idx+len(group)
            if done%200==0 or done==len(candidates):print(json.dumps({'split':args.split,'checkpoint':args.model.name,
                'questions':done,'seconds':round(time.perf_counter()-start,1)}),flush=True)
    if code!={p.relative_to(ROOT).as_posix():sha(p) for p in files}:raise ValueError('Scorer changed during run')
    write_json({'status':'completed','questions':len(candidates),'pairs':model.pairs_scored,
        'seconds':time.perf_counter()-start,'scores_sha256':sha(args.output/'scores.jsonl'),
        'protocol_sha256':sha(args.output/'protocol.json')},args.output/'completion.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--split',choices=['valid','test'],required=True);p.add_argument('--selection',type=Path)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
