"""Memory-bounded, conservative character-TFIDF audit against all accessible history.

The vocabulary contains every candidate character n-gram. Historical-only
dimensions are omitted from historical norms, making each cosine an UPPER
BOUND on its full-vocabulary counterpart. Excluding upper bounds >= .90 may
over-exclude, but cannot miss a >= .90 full-space match with this tokenization.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]
LOCAL=ROOT/'reports/source_holdout_v1/local'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--historical',type=Path,default=LOCAL/'historical_texts.jsonl')
    parser.add_argument('--output',type=Path,default=LOCAL/'overlap_audit.json');args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve previous audit; supply a fresh output path')
    started=time.perf_counter()
    pairs=json.loads((LOCAL/'extracted_pairs.json').read_text(encoding='utf8'))
    import sys
    sys.path.insert(0,str(ROOT))
    from scripts.prepare_source_faq import norm
    historical=[]; origins=[]
    with args.historical.open(encoding='utf8') as handle:
        for line in handle:
            r=json.loads(line);historical.append(r['text']);origins.append(r['origin'])
    candidates=[norm(r[k]) for r in pairs for k in ['question','answer']]
    vectorizer=CountVectorizer(analyzer='char_wb',ngram_range=(3,5),lowercase=False,dtype=np.float32)
    qc=vectorizer.fit_transform(candidates).tocsr()
    df=np.asarray((qc>0).sum(axis=0)).ravel().astype(np.float64)
    for start in range(0,len(historical),1024):
        matrix=vectorizer.transform(historical[start:start+1024])
        df+=np.asarray((matrix>0).sum(axis=0)).ravel()
        if start%32768==0:print(json.dumps({'phase':'idf','rows':start,'total':len(historical)}),flush=True)
    idf=(np.log((1+len(historical)+len(candidates))/(1+df))+1).astype(np.float32)
    q=normalize(qc.multiply(idf),norm='l2').tocsr()
    best=np.full(len(candidates),-1.,dtype=np.float32); indices=np.zeros(len(candidates),dtype=np.int64)
    for start in range(0,len(historical),1024):
        matrix=normalize(vectorizer.transform(historical[start:start+1024]).multiply(idf),norm='l2').tocsr()
        sims=(q @ matrix.T).toarray()
        local=sims.argmax(axis=1); maxima=sims[np.arange(len(candidates)),local]
        improve=maxima>best;best[improve]=maxima[improve];indices[improve]=start+local[improve]
        if start%32768==0:print(json.dumps({'phase':'similarity','rows':start,'total':len(historical)}),flush=True)
    exact=set(historical); rows=[]
    for i,r in enumerate(pairs):
        details={k:{'exact':candidates[2*i+j] in exact,'cosine_upper_bound':float(best[2*i+j]),
                    'historical_origin':origins[indices[2*i+j]],
                    'historical_text_sha256':hashlib.sha256(historical[indices[2*i+j]].encode()).hexdigest()}
                 for j,k in enumerate(['question','answer'])}
        rows.append({'id':r['id'],'publisher':r['publisher'],'excluded':any(d['exact'] or d['cosine_upper_bound']>=.9 for d in details.values()),**details})
    # Audit cross-publisher candidate pairs with the same conservative vector space.
    cross=(q @ q.T).toarray(); conflicts=[]
    for i,a in enumerate(pairs):
        for j,b in enumerate(pairs[:i]):
            if a['publisher']!=b['publisher']:
                sim=max(float(cross[2*i,2*j]),float(cross[2*i+1,2*j+1]))
                if sim>=.9:
                    conflicts.append({'first':a['id'],'second':b['id'],'similarity':sim})
                    rows[i]['excluded']=rows[j]['excluded']=True
    report={'historical_texts':len(historical),'candidate_pairs':len(pairs),'vocabulary':len(vectorizer.vocabulary_),
            'method':'char_wb 3-5, raw term counts, smoothed corpus IDF, cosine upper bound after omitting historical-only dimensions',
            'threshold':.9,'excluded':sum(r['excluded'] for r in rows),'cross_publisher_conflicts':conflicts,'rows':rows,
            'seconds':time.perf_counter()-started,'candidate_file_sha256':hashlib.sha256((LOCAL/'extracted_pairs.json').read_bytes()).hexdigest(),
            'historical_file_sha256':hashlib.sha256(args.historical.read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','cross_publisher_conflicts']},indent=2),flush=True)
if __name__=='__main__':main()
