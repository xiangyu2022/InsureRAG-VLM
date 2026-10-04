"""Remove every new held-out source from training negatives before validation."""
from collections import Counter
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import write_jsonl
from scripts.prepare_reranker_training import normalize
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.reranker import LocalCrossEncoder


def run():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4)
    old=ROOT/'data/training/retention_v1';out=ROOT/'data/training/retention_v2';out.mkdir(exist_ok=False)
    fixture=ROOT/'data/benchmarks/multidomain_v1'
    lock=json.loads((old/'manifest.lock.json').read_text(encoding='utf8'))
    for name,expected in lock['files'].items():assert sha(old/name)==expected
    answers=read_jsonl(old/'answers.jsonl');lookup={a['id']:a['text'] for a in answers};normal={a['id']:normalize(a['text']) for a in answers}
    held=read_jsonl(fixture/'valid.jsonl')+read_jsonl(fixture/'test.jsonl');sources={c['source_group'] for c in held}
    forbidden={a['id'] for a in answers if a.get('source_group') in sources}
    forbidden_text={normal[a] for a in forbidden};mask=np.array([a['id'] in forbidden or normal[a['id']] in forbidden_text for a in answers])
    forbidden.update(a['id'] for a,m in zip(answers,mask) if m)
    groups=read_jsonl(old/'train_groups.jsonl');changed=[g for g in groups if set(g['negative_ids'])&forbidden]
    affected_before=sum(len(set(g['negative_ids'])&forbidden) for g in groups)
    encoder=EmbeddingRetriever(str(ROOT/'../models/bge-small-en-v1.5'),use_hf_api=False,pooling='cls',max_length=512,
                              query_instruction='Represent this sentence for searching relevant passages: ')
    _,model=encoder._ensure_local_transformer();model.to('cuda')
    qv=np.vstack([encoder.embed_texts([encoder.query_instruction+g['question'] for g in changed[i:i+128]]) for i in range(0,len(changed),128)])
    dense=np.load(ROOT/'reports/multidomain_v1/index_cache/answer_embeddings.npy');sparse=SparseBM25([a['text'] for a in answers]);rng=np.random.default_rng(20261001)
    for i,g in enumerate(changed):
        d=dense@qv[i];b=sparse.scores(g['question']);d[mask]=-np.inf;b[mask]=0
        pos=set(g['positive_ids']);texts={normal[a] for a in pos};negs=[];ns={};seen=set()
        assert not pos&forbidden
        for source,order in [('dense_hard',ranked(d)),('lexical_hard',ranked(b,positive_only=True))]:
            count=0
            for j in order:
                aid=answers[j]['id'];value=normal[aid]
                if aid in forbidden or aid in pos or value in texts or value in seen:continue
                negs.append(aid);ns[aid]=source;seen.add(value);count+=1
                if count==4:break
            assert count==4
        for j in rng.permutation(len(answers)):
            aid=answers[j]['id'];value=normal[aid]
            if aid not in forbidden and aid not in pos and value not in texts and value not in seen:
                negs.append(aid);ns[aid]='random';break
        g['negative_ids']=negs;g['negative_sources']=ns
        if (i+1)%500==0:print(json.dumps({'source_isolated_groups':i+1}),flush=True)
    model.to('cpu');del model,encoder
    teacher=LocalCrossEncoder(ROOT/'../models/ms-marco-MiniLM-L6-v2','cuda',64,512)
    missing=[(g,aid) for g in changed for aid in g['negative_ids'] if aid not in g['teacher_scores']]
    for i in range(0,len(missing),256):
        batch=missing[i:i+256];scores=teacher.score_pairs([(g['question'],lookup[a]) for g,a in batch])
        for (g,a),score in zip(batch,scores):g['teacher_scores'][a]=float(score)
    for g in groups:
        assert not (set(g['positive_ids'])|set(g['negative_ids']))&forbidden
        g['teacher_scores']={a:g['teacher_scores'][a] for a in g['positive_ids']+g['negative_ids']}
    (out/'answers.jsonl').write_bytes((old/'answers.jsonl').read_bytes());write_jsonl(groups,out/'train_groups.jsonl')
    write_json({'excluded_source_groups':sorted(sources),'excluded_answer_candidates':len(forbidden),'remined_groups':len(changed),
                'removed_heldout_source_negative_pairs':affected_before,'remaining_heldout_source_positive_or_negative_pairs':0,
                'extra_teacher_pairs':teacher.pairs_scored,'policy':'All paragraphs from every new valid/test source excluded from positive and negative training pairs.'},out/'source_isolation.json')
    files=['answers.jsonl','train_groups.jsonl','source_isolation.json']
    write_json({**{k:v for k,v in lock.items() if k not in ['files','code_sha256','created_utc']},
                'created_utc':datetime.now(timezone.utc).isoformat(),'parent_retention_manifest_sha256':sha(old/'manifest.lock.json'),
                'teacher_pairs_in_final_training':sum(len(g['teacher_scores']) for g in groups),'extra_teacher_computations':teacher.pairs_scored,
                'source_isolation':'Strict for all new validation/test source groups, including negatives.',
                'code_sha256':sha(Path(__file__)),'files':{n:sha(out/n) for n in files}},out/'manifest.lock.json')
    print(json.dumps({'status':'completed','training_questions':len(groups),'remined_groups':len(changed),'extra_teacher_pairs':teacher.pairs_scored}),flush=True)

if __name__=='__main__':run()
