"""Exercise source-page navigation over real plan wording links; no QA claim."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import sha,read_jsonl,write_json,verify_fixture
from scripts.eval_insuranceqa_scale import SparseBM25,ranked
from scripts.build_policy_wordings import literal_references
from src.insurerag_vlm.graph import build_annotated_reference_edges,build_graph_adjacency,expand_candidate_page_keys


def run(args):
    manifest=json.loads((args.corpus/'manifest.json').read_text(encoding='utf8'))
    for name,digest in manifest['files'].items():
        if sha(args.corpus/name)!=digest:raise ValueError('Corpus mismatch: '+name)
    lock=verify_fixture(args.fixture)
    if lock['corpus_manifest_sha256']!=sha(args.corpus/'manifest.json'):raise ValueError('Wrong corpus')
    pages=read_jsonl(args.corpus/'rag_pages.jsonl')
    for row in pages:row['page_key']=f'{row["doc_id"]}::p{row["page"]:04d}'
    edges=build_annotated_reference_edges(pages);adjacency=build_graph_adjacency(edges)
    by_doc=defaultdict(list)
    for page in pages:by_doc[page['doc_id']].append(page)
    sparse={doc:SparseBM25([p['text'] for p in ps]) for doc,ps in by_doc.items()}
    printed_maps={}
    for doc,ps in by_doc.items():
        labels=defaultdict(list)
        for page in ps:
            if page.get('printed_page_label'):labels[page['printed_page_label']].append(page['page'])
        printed_maps[doc]=labels
    cases=read_jsonl(args.fixture/'cases.jsonl')
    args.output.mkdir(parents=True,exist_ok=False)
    codepaths=[Path(__file__),ROOT/'src/insurerag_vlm/graph.py',ROOT/'scripts/eval_insuranceqa_scale.py',ROOT/'scripts/build_policy_wordings.py']
    code={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in codepaths}
    write_json({'created_utc':datetime.now(timezone.utc).isoformat(),'fixture_lock_sha256':sha(args.fixture/'manifest.lock.json'),
        'code_sha256':code,'generation_calls':0,'seed':'Known anchor page is supplied to every arm',
        'candidate_scope':'Same explicitly identified plan brochure','budgets':{'context_pages':3,'graph_hops':2,'graph_expansions':8},
        'arms':['anchor_plus_bm25','anchor_plus_graph_then_bm25','direct_printed_page_lookup'],
        'limitations':lock['limitations']+['Direct lookup is included because following a literal page number does not require a graph.']},args.output/'protocol.json')
    results=[]
    with (args.output/'predictions.jsonl').open('w',encoding='utf8') as out:
        for case in cases:
            doc=case['document_scope'][0];ps=by_doc[doc]
            anchor=f'{doc}::p{case["anchor_page"]:04d}';target=f'{doc}::p{case["target_page"]:04d}'
            candidates=[ps[i]['page_key'] for i in ranked(sparse[doc].scores(case['anchor_text']),limit=len(ps),positive_only=True)]
            literal_targets,_=literal_references(case['anchor_text'],printed_maps[doc],case['anchor_page'])
            direct=[f'{doc}::p{ref["target"]:04d}' for ref in literal_targets]
            paths=expand_candidate_page_keys({anchor},adjacency,False,False,False,False,max_hops=2,max_expansions=8,explicit_only=True)
            for arm,order in [('anchor_plus_bm25',[anchor]+candidates),
                ('anchor_plus_graph_then_bm25',[anchor]+[p['page_key'] for p in paths]+candidates),
                ('direct_printed_page_lookup',[anchor]+direct+candidates)]:
                chosen=list(dict.fromkeys(order))[:3]
                row={'id':case['id'],'split':case['split'],'document_family_id':case['document_family_id'],
                    'arm':arm,'target_in_context':target in chosen,'selected_page_keys':chosen,'gold_target':target,
                    'paths':paths if 'graph' in arm else []}
                results.append(row);out.write(json.dumps(row)+'\n')
    summary={}
    for split in ['dev','test']:
        summary[split]={}
        for arm in ['anchor_plus_bm25','anchor_plus_graph_then_bm25','direct_printed_page_lookup']:
            rows=[r for r in results if r['split']==split and r['arm']==arm]
            summary[split][arm]={'n':len(rows),'target_in_context':sum(r['target_in_context'] for r in rows)/len(rows),
                'by_family':{family:{'n':len(group),'hits':sum(r['target_in_context'] for r in group)}
                    for family in sorted({r['document_family_id'] for r in rows})
                    if (group:=[r for r in rows if r['document_family_id']==family])}}
    if code!={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in codepaths}:raise ValueError('Code changed during evaluation')
    verify_fixture(args.fixture)
    write_json({'status':'completed','edges':len(edges),'cases':len(cases),'rankings':len(results),
                'generation_calls':0,'summaries':summary,'predictions_sha256':sha(args.output/'predictions.jsonl')},args.output/'summary.json')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=ROOT/'data/research_corpus/policy_wordings_v1')
    p.add_argument('--fixture',type=Path,default=ROOT/'data/benchmarks/opm_reference_navigation_v1')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
