"""Conservative source-span extraction for previously unused publisher Q/A formats."""
import re
from collections import Counter

MARKER=re.compile(r'(?mi)^[ \t]*(?:Question[ \t]*(?:\d{1,3}[.:)]|[-:])|Q(?:-[A-Z])?[ \t]*\d{0,3}[.:)])\s*')
ANS=re.compile(r'(?mi)^[ \t]*(?:Answer|A)[ \t]*\d{0,3}[.:)\-]\s*')
START=re.compile(r'(?i)^(?:yes\b|no\b|not\b|generally\b|the\b|states?\b|a\b|an\b|under\b|if\b|cms\b|as\b|for\b|in\b|section\b|effective\b|it\b|there\b|we\b|plans?\b|to\b|employers?\b|all\b|this\b|federal\b|individuals\b|children\b|medicaid\b|each\b|while\b|these\b|both\b|eligible\b|additional\b|beneficiaries\b)')
INTERROGATIVE=re.compile(r'(?i)^(?:what|when|where|why|how|who|which|can|could|does|do|did|is|are|will|would|must|should)\b')
clean=lambda text:' '.join(text.split())

def extract(text):
    pairs=[];rejections=Counter();markers=list(MARKER.finditer(text))
    for i,m in enumerate(markers):
        end=markers[i+1].start() if i+1<len(markers) else len(text);block=text[m.end():end]
        explicit=ANS.search(block)
        if explicit:
            qend=m.end()+explicit.start();astart=m.end()+explicit.end();method='explicit_qa_markers'
            # Numbers may be absent but cannot conflict when supplied.
            qnum=re.findall(r'\d+',m.group());anum=re.findall(r'\d+',explicit.group())
            if qnum and anum and qnum!=anum:rejections['number_mismatch']+=1;continue
        else:
            stop=re.search(r'\?[ \t]*\r?\n',block)
            if not stop:continue
            qend=m.end()+stop.start()+1;astart=m.end()+stop.end();method='numbered_question_terminated_line'
        q=clean(text[m.end():qend]);a=clean(text[astart:end])
        if not q.endswith('?') or not 5<=len(q.split())<=120:continue
        if not 25<=len(a.split())<=400:continue
        if not START.search(a) or INTERROGATIVE.search(a):continue
        if '?' in a[:180] or re.search(r'\b(?:previous question|described above|question \d+|Q\d+|these funds|this provision)\b|\[program|\{',q,re.I):continue
        pairs.append({'question':q,'text':a,'question_span':[m.end(),qend],'answer_span':[astart,end],'extraction_method':method})
    # Plain-paragraph FAQs need a publisher FAQ heading and both paragraph boundaries.
    if re.search(r'frequently\s+asked|questions\s+(?:and|&)\s+answers|\bFAQs?\b',text[:6000],re.I):
        paragraphs=list(re.finditer(r'\S[^\n]*(?:\n(?![ \t]*\n)[^\n]*)*',text))
        qs=[]
        for p in paragraphs:
            q=clean(p.group());q=re.sub(r'^\d+[.)]\s*','',q)
            if q.endswith('?') and 5<=len(q.split())<=100 and INTERROGATIVE.search(q):qs.append(p)
        for i,p in enumerate(qs):
            end=qs[i+1].start() if i+1<len(qs) else len(text);start=p.start()
            prefix=re.match(r'\d+[.)]\s*',text[start:p.end()])
            if prefix:start+=prefix.end()
            q=clean(text[start:p.end()]);astart=p.end()
            tag=re.match(r'\s*(?:Answer|A)[ \t]*\d{0,3}[.:)\-]\s*',text[astart:end],re.I)
            if tag:astart+=tag.end()
            a=clean(text[astart:end])
            if not 25<=len(a.split())<=400 or not re.match(r'[A-Z]',a) or INTERROGATIVE.search(a):continue
            if '?' in a[:180] or MARKER.search(text[p.end():end]):continue
            if re.search(r'\[program|\{|\b(?:previous question|these funds|this provision)\b',q,re.I):continue
            pairs.append({'question':q,'text':a,'question_span':[start,p.end()],'answer_span':[astart,end],'extraction_method':'publisher_faq_paragraphs'})
    unique={clean(p['question']).lower():p for p in pairs}
    return list(unique.values()),dict(rejections)
