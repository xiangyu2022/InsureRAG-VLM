"""Additional human/publisher questions and a fresh source-disjoint government test."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_multidomain_data import canonical_url,write_jsonl,digest
from scripts.prepare_reranker_training import normalize
from scripts.extract_additional_government import extract

FOCUS=re.compile(r'\b(?:how many|how much|what year|which year|when|before|after|during|between|at least|under|over|age|if|except|less than|more than)\b|\d',re.I)

def nearest(train,held):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2)
    x=vectorizer.fit_transform([c['question'] for c in held+train]);nn=NearestNeighbors(n_neighbors=1,metric='cosine',n_jobs=4).fit(x[:len(held)])
    result=[]
    for i in range(len(held),x.shape[0],128):
        d,j=nn.kneighbors(x[i:i+128]);result.extend((float(1-a[0]),held[int(b[0])]['id']) for a,b in zip(d,j))
    return result

def run():
    out=ROOT/'data/benchmarks/condition_v1';out.mkdir(parents=True,exist_ok=False)
    previous=ROOT/'data/benchmarks/multidomain_v1';allold=[c for s in ['train','valid','test'] for c in read_jsonl(previous/f'{s}.jsonl')]
    oldgov=read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/questions.jsonl')
    govold=[c for c in allold if c['domain']=='government']+oldgov
    usedurls={canonical_url(c['source_url']) for c in govold};trainsources={c['source_group'] for c in allold if c['split']=='train' and c['domain']=='government'}
    held=[c for c in allold if c['split']!='train']+oldgov
    held += read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/valid.jsonl')+read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/test.jsonl')
    oldgroups=read_jsonl(ROOT/'data/training/retention_v2/train_groups.jsonl')
    oldanswers=read_jsonl(ROOT/'data/training/retention_v2/answers.jsonl');answers={a['id']:a for a in oldanswers}
    records=read_jsonl(ROOT/'data/research_corpus/hicric_public_v1/records.jsonl')
    dol=read_jsonl(ROOT/'data/research_corpus/dol_additional_v1/records.jsonl');trainraw=[];testraw=[]
    excluded_records={'hicric_c6f1b7e8feddbf7e':'Previously observed severe repeated-glyph OCR','hicric_d061b462979df7a1':'Severe interleaved-glyph OCR observed before inference'}
    def add(pair,rec,source,split):
        aid='condgov_a_'+digest(normalize(pair['text']))[:16]
        answers.setdefault(aid,{'id':aid,'text':pair['text'],'source_group':source,'source_url':rec['source_url'],'domain':'government'})
        return {k:v for k,v in {**pair,'id':'condgov_q_'+digest(normalize(pair['question']))[:16],'gold_answer_ids':[aid],
                'source_record_id':rec['id'],'source_group':source,'source_url':rec['source_url'],'domain':'government','split':split,
                'annotation_origin':'Publisher-written Q/A, mechanical source-span extraction'}.items() if k!='text'}
    for rec in records:
        if rec['category']!='regulatory-guidance' or rec['id'] in excluded_records:continue
        urls={canonical_url(p['source_url']) for p in rec['provenance']};rec={**rec,'source_url':rec['provenance'][0]['source_url']}
        if urls&trainsources:split='train';source=sorted(urls&trainsources)[0]
        elif not urls&usedurls:split='test';source=sorted(urls)[0]
        else:continue
        pairs,_=extract(rec['text'])
        for pair in pairs:(trainraw if split=='train' else testraw).append(add(pair,rec,source,split))
    forbidden_parts={m.group(1) for c in held if 'source_url' in c for m in [re.search(r'aca-part-(\d+)',c['source_url'])] if m}
    for rec in dol:
        part=rec['source_url'].rsplit('-',1)[-1]
        if part in forbidden_parts:continue
        pairs,_=extract(rec['text'])
        trainraw.extend(add(p,rec,rec['source_group'],'train') for p in pairs)
    # Fresh test excludes any question previously trained or scored, plus answer texts seen in training pairs.
    priorquestions=allold+oldgov+read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/train.jsonl')+held
    seen_training_text={normalize(answers[a]['text']) for g in oldgroups for a in g['positive_ids']+g['negative_ids']}
    previous_question_keys={normalize(c['question']) for c in priorquestions};question_groups=defaultdict(list)
    for c in testraw:question_groups[normalize(c['question'])].append(c)
    tests=[];exclusions=[]
    for q,cs in question_groups.items():
        if q in previous_question_keys or len({normalize(answers[c['gold_answer_ids'][0]]['text']) for c in cs})!=1:
            exclusions.append({'id':cs[0]['id'],'reason':'previous_or_ambiguous_question'});continue
        c=sorted(cs,key=lambda c:c['source_group'])[0]
        if normalize(answers[c['gold_answer_ids'][0]]['text']) in seen_training_text:
            exclusions.append({'id':c['id'],'reason':'answer_text_seen_in_previous_training'});continue
        tests.append(c)
    similarities=nearest(tests,priorquestions)
    tests=[c for c,(score,qid) in zip(tests,similarities) if score<.92 or not exclusions.append({'id':c['id'],'reason':'near_previously_used_question','cosine':score,'nearest':qid}) and False]
    # Extra SQuAD examples come only from previously unused training paragraphs.
    used_paragraphs={a for c in allold if c['split']=='train' for a in c['gold_answer_ids']}
    payload=json.loads((ROOT/'../squad_download/squad_train_v1.1.json').read_text(encoding='utf8'));squad=[]
    for article in payload['data']:
        for para in article['paragraphs']:
            aid='squad_a_'+digest(normalize(para['context']))[:16]
            if aid not in answers or aid in used_paragraphs:continue
            qs=[q for q in para['qas'] if 5<=len(q['question'].split())<=60 and FOCUS.search(q['question'])
                and not re.search(r'\b(?:this passage|this paragraph|the passage|the paragraph)\b',q['question'],re.I)
                and all(para['context'][a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in q['answers'])]
            if not qs:continue
            q=min(qs,key=lambda q:digest(q['id']))
            squad.append({'id':'squad_q_'+q['id'],'question':q['question'],'gold_answer_ids':[aid],'domain':'general','source_group':'wikipedia:'+article['title'],
                'source_url':'https://en.wikipedia.org/wiki/'+article['title'],'split':'train','original_answer_spans':q['answers'],
                'annotation_origin':'Original SQuAD crowdworker; numeric/time/condition-oriented question from unused training paragraph'})
    trainraw += sorted(squad,key=lambda c:digest('condition-train|'+c['id']))[:6500]
    seen={normalize(g['question']) for g in oldgroups};heldq=held+tests
    heldtext={normalize(answers[a]['text']) for c in heldq for a in c['gold_answer_ids'] if a in answers}
    heldtext.update(normalize(a['text']) for a in read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/answers.jsonl'))
    newheldsources={c['source_group'] for c in tests}|{c['source_group'] for c in allold if c['split']!='train'}
    similarities=nearest(trainraw,heldq);train=[];general=0
    for c,(score,qid) in zip(trainraw,similarities):
        key=normalize(c['question']);reason=None
        if key in seen:reason='duplicate_training_question'
        elif score>=.92:reason='near_heldout_question'
        elif any(normalize(answers[a]['text']) in heldtext for a in c['gold_answer_ids']):reason='heldout_positive_text'
        elif c['source_group'] in newheldsources:reason='heldout_source'
        if reason:exclusions.append({'id':c['id'],'reason':reason,'nearest':qid,'cosine':score});continue
        seen.add(key)
        if c['domain']=='general':
            if general>=6000:continue
            general+=1
        train.append(c)
    # Keep only referenced new government answer texts; preserve the old complete index unchanged.
    newids={a for c in train+tests for a in c['gold_answer_ids']};oldids={a['id'] for a in oldanswers}
    answerrows=oldanswers+[answers[a] for a in sorted(newids-oldids)]
    write_jsonl(sorted(train,key=lambda c:c['id']),out/'train_additions.jsonl');write_jsonl(sorted(tests,key=lambda c:c['id']),out/'test.jsonl');write_jsonl(answerrows,out/'answers.jsonl')
    write_json({'excluded_records':excluded_records,'exclusions':exclusions,'fresh_test_questions_compared_against_all_previous_questions':True,
        'fresh_test_positive_text_compared_against_all_previous_training_pairs':True,'no_model_metrics_used':True},out/'audit.json')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'train_additions':dict(Counter(c['domain'] for c in train)),
        'fresh_test_questions':len(tests),'fresh_test_sources':len({c['source_group'] for c in tests}),'answer_candidates':len(answerrows),
        'source_group_policy':'Fresh test uses only previously unused government URLs; all of their answer texts excluded from future training pairs.',
        'frozen_before_model_inference':True,'previous_training_manifest_sha256':sha(ROOT/'data/training/retention_v2/manifest.lock.json'),
        'previous_fixture_sha256':sha(previous/'manifest.lock.json'),'dol_source_manifest_sha256':sha(ROOT/'data/research_corpus/dol_additional_v1/manifest.json'),
        'code_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/extract_additional_government.py']},
        'files':{n:sha(out/n) for n in ['train_additions.jsonl','test.jsonl','answers.jsonl','audit.json']},
        'limitations':['Mechanical Q/A extraction, not expert adjudication.','Government texts retain source dates; not present-day legal advice.','Existing tests are historical regressions this iteration.','No new independent general-language test; previous general test is historical.']}
    write_json(manifest,out/'manifest.lock.json');print(json.dumps({k:manifest[k] for k in ['train_additions','fresh_test_questions','fresh_test_sources','answer_candidates']},indent=2))

if __name__=='__main__':run()
