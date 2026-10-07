"""Memory-bounded lexical near-duplicate screening, before any model evaluation.

Character cosine uses candidate vocabulary with historical-only dimensions
omitted, a conservative upper bound. Flags are exclusions/holds, never a claim
of semantic novelty. Prior exposed source FAQs are included explicitly.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.holdout_quality import normalize as norm
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--history',type=Path,required=True)
    p.add_argument('--prior-faq',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threshold',type=float,default=.9)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous audit')
    if not .8<=a.threshold<=1:raise ValueError('Threshold outside declared screening range')
    started=time.perf_counter();rows=[json.loads(l) for l in a.candidates.read_text(encoding='utf8').splitlines() if l.strip()]
    candidates=[norm(r[k]) for r in rows for k in ['question','answer']]
    texts=[];origins=[]
    with a.history.open(encoding='utf8') as handle:
        for line in handle:
            r=json.loads(line);texts.append(norm(r['text']));origins.append(r['origin'])
    prior=json.loads(a.prior_faq.read_text(encoding='utf8'))
    for r in prior:
        for key in ['question','answer']:
            texts.append(norm(r[key]));origins.append('prior_exposed_faq/'+r['id']+'/'+key)
    vectorizer=CountVectorizer(analyzer='char_wb',ngram_range=(3,5),lowercase=False,dtype=np.float32)
    qc=vectorizer.fit_transform(candidates).tocsr();df=np.asarray((qc>0).sum(axis=0)).ravel().astype(np.float64)
    for start in range(0,len(texts),1024):
        matrix=vectorizer.transform(texts[start:start+1024]);df+=np.asarray((matrix>0).sum(axis=0)).ravel()
        if start%65536==0:print(json.dumps({'phase':'idf','rows':start,'total':len(texts)}),flush=True)
    idf=(np.log((1+len(texts)+len(candidates))/(1+df))+1).astype(np.float32)
    q=normalize(qc.multiply(idf),norm='l2').tocsr();best=np.zeros(len(candidates),dtype=np.float32);indices=np.zeros(len(candidates),dtype=np.int64)
    for start in range(0,len(texts),512):
        matrix=normalize(vectorizer.transform(texts[start:start+512]).multiply(idf),norm='l2').tocsr()
        sims=(q @ matrix.T).toarray();local=sims.argmax(axis=1);maxima=sims[np.arange(len(candidates)),local]
        improve=maxima>best;best[improve]=maxima[improve];indices[improve]=start+local[improve]
        if start%65536==0:print(json.dumps({'phase':'history_similarity','rows':start,'total':len(texts)}),flush=True)
    exact=set(texts);screen=[]
    for i,row in enumerate(rows):
        details={key:{'exact':candidates[2*i+j] in exact,'cosine_upper_bound':float(best[2*i+j]),
                    'origin':origins[indices[2*i+j]],'historical_text_sha256':hashlib.sha256(texts[indices[2*i+j]].encode()).hexdigest()}
                 for j,key in enumerate(['question','answer'])}
        screen.append({'id':row['id'],'publisher':row['publisher'],**details,'historical_flag':any(v['exact'] or v['cosine_upper_bound']>=a.threshold for v in details.values())})
    pairs=[]
    for start in range(0,len(rows),256):
        for kind,offset in [('question',0),('answer',1)]:
            sim=(q[2*start+offset:2*min(start+256,len(rows)):2] @ q[offset::2].T).toarray()
            for ii,jj in zip(*np.where(sim>=.86)):
                i=start+int(ii);j=int(jj)
                if i<=j:continue
                pairs.append({'first':rows[i]['id'],'second':rows[j]['id'],'field':kind,'similarity':float(sim[ii,j]),
                              'exact':candidates[2*i+offset]==candidates[2*j+offset],
                              'same_publisher':rows[i]['publisher']==rows[j]['publisher'],
                              'same_document':rows[i]['document_group']==rows[j]['document_group']})
    report={'candidate_count':len(rows),'historical_strings_including_exposed_faq':len(texts),'prior_exposed_faq_pairs':len(prior),
      'method':'NFKC/casefold word normalization; char_wb 3-5 TFIDF conservative historical cosine upper bound',
      'historical_flag_threshold':a.threshold,'candidate_pair_review_threshold':.86,'historical_flagged':sum(r['historical_flag'] for r in screen),
      'candidate_similarity_edges':len(pairs),'candidate_pair_flags':pairs,'rows':screen,
      'input_sha256':{'candidates':sha(a.candidates),'history':sha(a.history),'prior_faq':sha(a.prior_faq)},
      'seconds':time.perf_counter()-started,'accepted_test_items':0,
      'limitations':['Lexical screening does not establish semantic independence.','Historical raw PDFs and denied paths are not reconstructed.','Document aliases/containment and foundation pretraining require separate caveats.']}
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'candidate_pair_flags','rows'}},indent=2))
if __name__=='__main__':main()
