"""Build mixed-domain hard negatives and cache a frozen public teacher's scores."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_reranker_training import normalize
from scripts.prepare_multidomain_data import write_jsonl
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from src.insurerag_vlm.retriever import EmbeddingRetriever
from src.insurerag_vlm.reranker import LocalCrossEncoder


def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4);threadpool_limits(4)
    fixture=ROOT/'data/benchmarks/multidomain_v1';oldtrain=ROOT/'data/training/insuranceqa_hardneg_v2'
    for directory in [fixture,oldtrain]:
        lock=json.loads((directory/'manifest.lock.json').read_text(encoding='utf8'))
        for name,expected in lock['files'].items():
            if sha(directory/name)!=expected:raise ValueError('Input fixture changed')
    args.output.mkdir(parents=True,exist_ok=False);args.index.mkdir(parents=True,exist_ok=False)
    legacy=sorted(read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/answers.jsonl'),key=lambda a:int(a['id']))
    added=read_jsonl(fixture/'answers.jsonl');answers=legacy+added;answer_map={a['id']:a['text'] for a in answers}
    if len(answer_map)!=len(answers):raise ValueError('Answer ID collision')
    original=read_jsonl(oldtrain/'train_groups.jsonl');newcases=read_jsonl(fixture/'train.jsonl')
    held=read_jsonl(fixture/'valid.jsonl')+read_jsonl(fixture/'test.jsonl')
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2)
    matrix=vectorizer.fit_transform([c['question'] for c in held+original])
    distances,indices=NearestNeighbors(n_neighbors=1,metric='cosine',n_jobs=4).fit(matrix[:len(held)]).kneighbors(matrix[len(held):])
    heldtexts={normalize(answer_map[a]) for c in held for a in c['gold_answer_ids']}
    groups=[];excluded=[]
    for g,d,j in zip(original,distances,indices):
        if 1-d[0]>=.92 or any(normalize(answer_map[a]) in heldtexts for a in g['positive_ids']):
            excluded.append({'id':g['id'],'cosine':float(1-d[0]),'nearest_heldout':held[int(j[0])]['id']});continue
        groups.append({**g,'source_domain':'insuranceqa'})
    write_json({'excluded_legacy_train':excluded,'heldout_used_only_for_exclusion':True},args.output/'split_hygiene.json')
    encoder=EmbeddingRetriever(str(args.embedding_model),use_hf_api=False,pooling='cls',max_length=512,
                               query_instruction='Represent this sentence for searching relevant passages: ')
    _,model=encoder._ensure_local_transformer();model.to('cuda')
    oldprotocol=json.loads((ROOT/'reports/insuranceqa_v2/rerank_valid_v2/protocol.json').read_text(encoding='utf8'))
    oldvectors=ROOT/'reports/insuranceqa_v2/retrieval_frozen/answer_embeddings.npy'
    if sha(oldvectors)!=oldprotocol['embedding_cache_sha256']['answer_embeddings.npy']:raise ValueError('Old embedding cache changed')
    strip=lambda d:{k:v for k,v in d.items() if k!='download_provenance.json'}
    if strip(encoder.index_fingerprint()['checkpoint_files_sha256'])!=strip(oldprotocol['embedding']['checkpoint_files_sha256']):raise ValueError('Embedding model changed')
    vectors=[];start=time.perf_counter()
    for i in range(0,len(added),256):
        vectors.append(encoder.embed_texts([a['text'] for a in added[i:i+256]]))
        if i%4096==0:print(json.dumps({'new_answer_embeddings':min(i+256,len(added)),'total':len(added)}),flush=True)
    dense=np.vstack([np.load(oldvectors),*vectors]);np.save(args.index/'answer_embeddings.npy',dense)
    queryvectors=[]
    for i in range(0,len(newcases),256):
        queryvectors.append(encoder.embed_texts([encoder.query_instruction+c['question'] for c in newcases[i:i+256]]))
    queryvectors=np.vstack(queryvectors)
    sparse=SparseBM25([a['text'] for a in answers]);normal={a['id']:normalize(a['text']) for a in answers};rng=np.random.default_rng(20261001)
    for i,c in enumerate(newcases):
        d=dense@queryvectors[i];b=sparse.scores(c['question']);positive=set(c['gold_answer_ids']);texts={normal[a] for a in positive}
        negatives=[];sources={};seen=set()
        for source,order in [('dense_hard',ranked(d)),('lexical_hard',ranked(b,positive_only=True))]:
            count=0
            for j in order:
                aid=answers[j]['id'];value=normal[aid]
                if aid in positive or value in texts or value in seen:continue
                negatives.append(aid);sources[aid]=source;seen.add(value);count+=1
                if count==4:break
        for j in rng.permutation(len(answers)):
            aid=answers[j]['id'];value=normal[aid]
            if aid not in positive and value not in texts and value not in seen:
                negatives.append(aid);sources[aid]='random';break
        groups.append({'id':c['id'],'question':c['question'],'positive_ids':c['gold_answer_ids'],
            'negative_ids':negatives,'negative_sources':sources,'source_domain':c['domain'],
            'source_group':c['source_group'],'domain':c['domain']})
        if (i+1)%1000==0:print(json.dumps({'new_training_groups_mined':i+1}),flush=True)
    write_jsonl(answers,args.output/'answers.jsonl');write_jsonl(groups,args.output/'train_groups_unscored.jsonl')
    model.to('cpu');del encoder,model
    teacher=LocalCrossEncoder(args.teacher,'cuda',64,512)
    with (args.output/'train_groups.jsonl').open('w',encoding='utf8') as handle:
        for start_index in range(0,len(groups),16):
            batch=groups[start_index:start_index+16];pairs=[]
            for g in batch:pairs.extend((g['question'],answer_map[a]) for a in g['positive_ids']+g['negative_ids'])
            scores=teacher.score_pairs(pairs);offset=0
            for g in batch:
                ids=g['positive_ids']+g['negative_ids'];g['teacher_scores']={a:float(v) for a,v in zip(ids,scores[offset:offset+len(ids)])};offset+=len(ids)
                handle.write(json.dumps(g)+'\n')
            handle.flush()
            if (start_index+len(batch))%1000==0 or start_index+len(batch)==len(groups):
                print(json.dumps({'teacher_scored_groups':start_index+len(batch),'total':len(groups),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    files=['answers.jsonl','train_groups.jsonl','train_groups_unscored.jsonl','split_hygiene.json']
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'source_fixture_sha256':sha(fixture/'manifest.lock.json'),
        'parent_training_manifest_sha256':sha(oldtrain/'manifest.lock.json'),'training_questions':len(groups),
        'source_domains':dict(Counter(g['source_domain'] for g in groups)),
        'available_positive_pairs':sum(len(g['positive_ids']) for g in groups),'mined_negative_pairs':sum(len(g['negative_ids']) for g in groups),
        'teacher':teacher.fingerprint(),'teacher_pairs':teacher.pairs_scored,'code_sha256':sha(Path(__file__)),
        'files':{name:sha(args.output/name) for name in files},
        'limitations':['New negative candidates are unlabeled, not expert-verified irrelevant.','Teacher scores are frozen public-model predictions, not human labels.',
                       'Government training uses URL-disjoint documents from every government evaluation.']}
    write_json(manifest,args.output/'manifest.lock.json')
    index={'created_utc':datetime.now(timezone.utc).isoformat(),'answer_count':len(answers),
        'answer_file_sha256':sha(args.output/'answers.jsonl'),'answer_embeddings_sha256':sha(args.index/'answer_embeddings.npy'),
        'embedding':oldprotocol['embedding'],'fixture_sha256':sha(fixture/'manifest.lock.json'),
        'code_sha256':sha(Path(__file__)),'status':'completed'}
    write_json(index,args.index/'manifest.json')
    print(json.dumps({k:manifest[k] for k in ['training_questions','source_domains','available_positive_pairs','mined_negative_pairs','teacher_pairs']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--embedding-model',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True)
    run(p.parse_args())
