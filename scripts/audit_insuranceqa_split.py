from pathlib import Path
import json,re,sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
fixture=ROOT/'data/benchmarks/insuranceqa_v2'
test=read_jsonl(fixture/'test.jsonl');other=read_jsonl(fixture/'train.jsonl')+read_jsonl(fixture/'valid.jsonl')
vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2)
allx=vec.fit_transform([r['question'] for r in other+test])
neighbors=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=2).fit(allx[:len(other)])
dist,ix=neighbors.kneighbors(allx[len(other):])
near=[dict(id=row['id'],question=row['question'],neighbor_id=other[int(ix[i,0])]['id'],neighbor_question=other[int(ix[i,0])]['question'],cosine=float(1-dist[i,0])) for i,row in enumerate(test) if 1-dist[i,0]>=.92]
parent=list(range(len(test)))
def find(i):
 while parent[i]!=i:
  parent[i]=parent[parent[i]];i=parent[i]
 return i
seen={}
for i,row in enumerate(test):
 for a in row['gold_answer_ids']:
  if a in seen:parent[find(i)]=find(seen[a])
  else:seen[a]=i
groups={}
for i in range(len(test)):groups.setdefault(find(i),[]).append(i)
groups=list(groups.values());rng=np.random.default_rng(20261001)
run=ROOT/'reports/insuranceqa_v2/retrieval_frozen'
preds=read_jsonl(run/'predictions.jsonl')
lookup={(r['id'],r['arm']):r for r in preds if r['split']=='test' and r['scope']=='all_27413_answers'}
nearids={r['id'] for r in near};summary={}
arrays={arm:np.array([lookup[(r['id'],arm)]['hit_at_10'] for r in test]) for arm in ['bm25','bge','rrf']}
draws={arm:[] for arm in arrays};paired=[]
sizes=np.array([len(g) for g in groups]);sums={arm:np.array([arr[g].sum() for g in groups]) for arm,arr in arrays.items()}
for _ in range(3000):
 selected=rng.integers(0,len(groups),len(groups));n=sizes[selected].sum()
 for arm in arrays:draws[arm].append(float(sums[arm][selected].sum()/n))
 paired.append(draws['bge'][-1]-draws['rrf'][-1])
for arm,arr in arrays.items():
 mask=np.array([r['id'] not in nearids for r in test])
 summary[arm]={'hit_at_10':float(arr.mean()),'shared_gold_cluster_bootstrap_95':np.quantile(draws[arm],[.025,.975]).tolist(),
               'char_ngram_near_duplicates_excluded_n':int(mask.sum()),'hit_at_10_after_excluding_similar_questions':float(arr[mask].mean())}
report={'status':'posthoc_sensitivity_audit_not_model_selection','test_questions':len(test),
 'near_duplicate_rule':'char_wb TF-IDF 3-5 grams, min_df=2; nearest train/valid cosine >= 0.92. Lexical proxy, not semantic deduplication.',
 'near_duplicate_count':len(near),'near_duplicates':near,'shared_gold_connected_components':len(groups),'largest_component':max(map(len,groups)),
 'bootstrap_seed':20261001,'bootstrap_draws':3000,'all_answer_retrieval':summary,
 'bge_minus_rrf_hit10_paired_cluster_95':np.quantile(paired,[.025,.975]).tolist(),
 'limitations':['Original test split and all main scores are retained. This is a sensitivity slice chosen after the run, not a new benchmark.',
                'Answer-ID clusters address shared labels only; remaining topical/paraphrase dependence and model pretraining exposure are not ruled out.']}
write_json(report,run/'sensitivity_audit.json')
print(json.dumps({k:v for k,v in report.items() if k!='near_duplicates'},indent=2))
