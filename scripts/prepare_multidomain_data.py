"""Source-held-out government Q/A plus original SQuAD paragraph retrieval data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_insuranceqa import read_jsonl,write_json,sha
from scripts.prepare_reranker_training import normalize

QUESTION=re.compile(r'(?mi)^[ \t]*Q(?:uestion)?[ \t]*(\d{0,3})[.:)][ \t]*')
ANSWER=re.compile(r'(?mi)^[ \t]*A(?:nswer)?[ \t]*(\d{0,3})[.:)][ \t]*')
clean=lambda s:' '.join(s.split())
digest=lambda s:hashlib.sha256(s.encode('utf8')).hexdigest()


def canonical_url(url):
    if 'web.archive.org/web/' in url:
        url=re.sub(r'^https?://web\.archive\.org/web/[^/]+/','',url)
    parsed=urlparse(url)
    return parsed.netloc.lower().removeprefix('www.')+parsed.path.lower().rstrip('/')


def extract_government(text):
    matches=list(QUESTION.finditer(text));pairs=[];rejected=Counter()
    has_faq_title=bool(re.search(r'frequently\s+asked|questions\s+(?:and|&)\s+answers|\bFAQs?\b',text[:6000],re.I))
    for i,m in enumerate(matches):
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        block=text[m.end():end];tag=ANSWER.search(block)
        if tag:
            if tag.group(1) and m.group(1) and tag.group(1)!=m.group(1):rejected['answer_number_mismatch']+=1;continue
            qend=m.end()+tag.start();astart=m.end()+tag.end();method='explicit_answer_marker'
        else:
            if not has_faq_title:rejected['no_answer_marker_or_faq_title']+=1;continue
            gap=re.search(r'\n[ \t]*\n',block)
            if not gap:rejected['no_question_paragraph_boundary']+=1;continue
            qend=m.end()+gap.start();astart=m.end()+gap.end();method='publisher_faq_paragraph_boundary'
        q=clean(text[m.end():qend]);a=clean(text[astart:end])
        if not q.endswith('?') or not 5<=len(q.split())<=120:rejected['question_shape']+=1;continue
        if not 25<=len(a.split())<=350:rejected['answer_length']+=1;continue
        if re.search(r'\b(?:previous question|question\s+\d+|Q\d+|described above|discussed above|listed above)\b|\[program|\{',q,re.I):
            rejected['context_dependent_question']+=1;continue
        # Reject another apparent question heading embedded inside a proposed answer.
        if re.search(r'\bQuestion\s+\d+[:.]',a):rejected['embedded_question_heading']+=1;continue
        pairs.append({'question':q,'text':a,'question_span':[m.end(),qend],'answer_span':[astart,end],
                      'extraction_method':method})
    return pairs,dict(rejected)


def write_jsonl(rows,path):
    with path.open('w',encoding='utf8') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False)+'\n')


def run(args):
    import requests
    args.output.mkdir(parents=True,exist_ok=False);args.cache.mkdir(parents=True,exist_ok=True)
    records=read_jsonl(ROOT/'data/research_corpus/hicric_public_v1/records.jsonl')
    previous=read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/questions.jsonl')
    excluded_urls={canonical_url(c['source_url']) for c in previous}
    quality_excluded={'hicric_c6f1b7e8feddbf7e':'Observed severe repeated-glyph OCR in question and answer text before model inference'}
    government=[];extractions=[]
    for rec in records:
        urls={canonical_url(p['source_url']) for p in rec['provenance']}
        if rec['category']!='regulatory-guidance' or urls&excluded_urls or rec['id'] in quality_excluded:continue
        pairs,rejections=extract_government(rec['text'])
        extractions.append({'record_id':rec['id'],'accepted':len(pairs),'rejections':rejections})
        for pair in pairs:
            source=sorted(urls)[0];question_key=normalize(pair['question'])
            government.append({**pair,'id':'gov2_q_'+digest(question_key)[:16],
                'answer_id':'gov2_a_'+digest(normalize(pair['text']))[:16],
                'source_group':source,'source_record_id':rec['id'],'source_url':rec['provenance'][0]['source_url'],
                'domain':'government','annotation_origin':'Publisher Q/A, mechanically extracted'})
    byquestion=defaultdict(list)
    for c in government:byquestion[normalize(c['question'])].append(c)
    government=[];ambiguous=[]
    old_questions={normalize(c['question']) for c in previous}
    old_answers={normalize(a['text']) for a in read_jsonl(ROOT/'data/benchmarks/hicric_government_qa_v1/answers.jsonl')}
    for key,group in byquestion.items():
        if len({normalize(c['text']) for c in group})>1 or key in old_questions:
            ambiguous.append({'question':group[0]['question'],'reason':'ambiguous_or_legacy_question'});continue
        if normalize(group[0]['text']) in old_answers:continue
        government.append(sorted(group,key=lambda c:c['source_group'])[0])
    sources=sorted({c['source_group'] for c in government},key=lambda x:digest('gov-split-20261001|'+x))
    cut1=round(len(sources)*.5);cut2=round(len(sources)*.75)
    partition={s:('train' if i<cut1 else 'valid' if i<cut2 else 'test') for i,s in enumerate(sources)}
    cases=[];answers={};answer_sources=defaultdict(set)
    for c in government:
        split=partition[c['source_group']];aid=c.pop('answer_id');text=c.pop('text')
        answers.setdefault(aid,{'id':aid,'text':text,'source_group':c['source_group'],'source_url':c['source_url'],'domain':'government'})
        c.update({'split':split,'gold_answer_ids':[aid]});cases.append(c);answer_sources[aid].add(split)
    # No normalized answer text may be a training positive if used by a held-out case.
    overlap={a for a,splits in answer_sources.items() if len(splits)>1}
    cases=[c for c in cases if not (set(c['gold_answer_ids'])&overlap)]
    upstream={};squad_candidates={'train':[],'dev':[]};titles={}
    for split in ['train','dev']:
        url=f'https://rajpurkar.github.io/SQuAD-explorer/dataset/{split}-v1.1.json';path=args.cache/f'squad_{split}_v1.1.json'
        if not path.exists():
            response=requests.get(url,timeout=90);response.raise_for_status();path.write_bytes(response.content)
        payload=json.loads(path.read_text(encoding='utf8'));upstream[split]={'url':url,'sha256':sha(path),'version':payload['version']}
        titles[split]={article['title'] for article in payload['data']}
        for article in payload['data']:
            title=article['title'];source='wikipedia:'+title
            for para in article['paragraphs']:
                text=para['context']
                if not 40<=len(text.split())<=300:continue
                aid='squad_a_'+digest(normalize(text))[:16]
                answers.setdefault(aid,{'id':aid,'text':text,'source_group':source,'source_url':'https://en.wikipedia.org/wiki/'+title,'domain':'general'})
                valid=[]
                for q in para['qas']:
                    if not 5<=len(q['question'].split())<=60:continue
                    if re.search(r'\b(?:this passage|this paragraph|the paragraph|the passage)\b',q['question'],re.I):continue
                    if not all(text[a['answer_start']:a['answer_start']+len(a['text'])]==a['text'] for a in q['answers']):continue
                    valid.append({'id':'squad_q_'+q['id'],'question':q['question'],'gold_answer_ids':[aid],
                        'source_group':source,'source_url':'https://en.wikipedia.org/wiki/'+title,'domain':'general',
                        'annotation_origin':'Original SQuAD crowdworker question; gold paragraph derived from validated answer span',
                        'original_answer_spans':q['answers']})
                # At most one question per paragraph, selected without any model scores.
                if valid:squad_candidates[split].append(min(valid,key=lambda q:digest(q['id'])))
    if titles['train']&titles['dev']:raise ValueError('SQuAD official title splits overlap')
    dev_titles=sorted({c['source_group'] for c in squad_candidates['dev']},key=lambda x:digest('squad-split-20261001|'+x))
    validation_titles=set(dev_titles[:len(dev_titles)//2])
    squad_train=sorted(squad_candidates['train'],key=lambda q:digest('sample|'+q['id']))[:args.general_train]
    squad_valid=sorted([c for c in squad_candidates['dev'] if c['source_group'] in validation_titles],key=lambda c:digest(c['id']))[:400]
    squad_test=sorted([c for c in squad_candidates['dev'] if c['source_group'] not in validation_titles],key=lambda c:digest(c['id']))[:600]
    for split,group in [('train',squad_train),('valid',squad_valid),('test',squad_test)]:
        cases.extend({**c,'split':split} for c in group)
    # Global hygiene: remove duplicate/near held-out questions and positive texts from new training.
    oldheld=read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/valid.jsonl')+read_jsonl(ROOT/'data/benchmarks/insuranceqa_v2/test.jsonl')+previous
    held=[c for c in cases if c['split']!='train']+oldheld
    held_ids={a for c in cases if c['split']!='train' for a in c['gold_answer_ids']}
    held_texts={normalize(answers[a]['text']) for a in held_ids}
    train=[c for c in cases if c['split']=='train'];seen=set();exclusions=[];retained=[]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2)
    matrix=vec.fit_transform([c['question'] for c in held+train])
    nn=NearestNeighbors(n_neighbors=1,metric='cosine',n_jobs=4).fit(matrix[:len(held)])
    distances,indices=nn.kneighbors(matrix[len(held):])
    for c,d,j in zip(train,distances,indices):
        key=normalize(c['question']);reason=None
        if key in seen:reason='duplicate_new_train_question'
        elif 1-d[0]>=.92:reason='near_heldout_question'
        elif any(normalize(answers[a]['text']) in held_texts for a in c['gold_answer_ids']):reason='heldout_positive_text'
        seen.add(key)
        if reason:exclusions.append({'id':c['id'],'reason':reason,'nearest_heldout':held[int(j[0])]['id'],'cosine':float(1-d[0])})
        else:retained.append(c)
    cases=[c for c in cases if c['split']!='train']+retained
    for split in ['train','valid','test']:
        write_jsonl(sorted([c for c in cases if c['split']==split],key=lambda c:c['id']),args.output/f'{split}.jsonl')
    write_jsonl(sorted(answers.values(),key=lambda a:a['id']),args.output/'answers.jsonl')
    write_json({'quality_excluded_records':quality_excluded,'government_extraction':extractions,'ambiguous':ambiguous,
        'cross_split_answer_ids_excluded':sorted(overlap),'new_training_exclusions':exclusions,
        'legacy_government_source_urls_excluded':sorted(excluded_urls)},args.output/'audit.json')
    files=['train.jsonl','valid.jsonl','test.jsonl','answers.jsonl','audit.json']
    counts={split:dict(Counter(c['domain'] for c in cases if c['split']==split)) for split in ['train','valid','test']}
    groups={split:{domain:len({c['source_group'] for c in cases if c['split']==split and c['domain']==domain}) for domain in ['government','general']} for split in counts}
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'counts':counts,'source_groups':groups,
        'answer_candidates':len(answers),'squad_upstream':upstream,'government_upstream_manifest_sha256':sha(ROOT/'data/research_corpus/hicric_public_v1/manifest.json'),
        'legacy_government_test_manifest_sha256':sha(ROOT/'data/benchmarks/hicric_government_qa_v1/manifest.lock.json'),
        'split_unit':'canonical government URL or Wikipedia article title; new train/valid/test source groups disjoint',
        'government_split_seed':'gov-split-20261001','squad_split_seed':'squad-split-20261001',
        'frozen_before_inference':True,'extraction_code_sha256':sha(Path(__file__)),
        'licenses':['SQuAD CC BY-SA 4.0 with original Wikipedia attribution','HICRIC CC BY-SA 4.0 with original government source attribution'],
        'limitations':['Derived paragraph/answer retrieval, not official SQuAD span EM/F1.','Public pretrained-model exposure unknown.',
            'Government mechanical extraction is not expert adjudication.','Source disjointness is not publisher/topic disjointness.'],
        'files':{name:sha(args.output/name) for name in files}}
    write_json(manifest,args.output/'manifest.lock.json');print(json.dumps({k:manifest[k] for k in ['counts','source_groups','answer_candidates']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--general-train',type=int,default=8000)
    run(p.parse_args())
