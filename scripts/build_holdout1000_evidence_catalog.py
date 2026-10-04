"""Bind complete paragraphs, list items and tables to immutable source text.

Short list facts and table headers must remain available. Catalog blocks are
source material, never approved benchmark items. Section spans preserve adjacent
qualifiers rather than selecting scattered numeric sentences.
"""
import argparse,hashlib,json,re,sys
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.insurerag_vlm.source_faq_extraction import clean,content_root

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def catalog_document(doc,raw):
 main=content_root(BeautifulSoup(raw,'html.parser'));headings=[];blocks=[];seen=set();cursor=0
 for node in main.find_all(['h1','h2','h3','h4','h5','h6','p','li','dd','table','va-table']):
  text=clean(node.get_text(' ',strip=True))
  if not text:continue
  if node.name.startswith('h'):
   level=int(node.name[1]);headings=[h for h in headings if h[0]<level]+[(level,text)]
   continue
  if any(node.find_parent(t) for t in ['p','li','dd','table','va-table']):continue
  if len(text)<25 or len(text)>(16000 if node.name in ['table','va-table'] else 6000) or text in seen:continue
  start=doc['text'].find(text,cursor)
  if start<0:start=doc['text'].find(text)
  if start<0:continue
  cursor=start+len(text);seen.add(text)
  flags=[]
  if re.search(r'NAIC|National Association of Insurance Commissioners|reproduced (?:with|by)|copyright',text,re.I):flags.append('third_party_rights_review')
  nums=re.findall(r'(?<!\w)(?:\$\s*)?\d[\d,.]*(?:\s*%)?',text)
  blocks.append({'block_id':digest(doc['id']+'|'+str(start)+'|'+text)[:20],'start':start,'end':start+len(text),
   'sha256':digest(text),'heading':headings[-1][1] if headings else '', 'heading_path':[h[1] for h in headings],
   'element':node.name,'table_title':node.get('table-title') if node.name=='va-table' else None,'text':text,'numeric_candidate':bool(nums and re.search(r'deductib|percent|maximum|minimum|benefit|coverage|premium|payment|cost|interest|rate|limit|days|months',text,re.I)),
   'numeric_token_count':len(nums),'rights_flags':flags})
 sections=[]
 for block in blocks:
  path=block['heading_path']
  if not sections or sections[-1]['heading_path']!=path:
   sections.append({'heading_path':path,'start':block['start'],'end':block['end'],'block_ids':[block['block_id']]})
  else:
   sections[-1]['end']=block['end'];sections[-1]['block_ids'].append(block['block_id'])
 for section in sections:
  text=doc['text'][section['start']:section['end']]
  section.update({'text':text,'sha256':digest(text),'section_id':digest(doc['id']+'|'+str(section['start'])+'|'+text)[:20]})
 return {k:doc[k] for k in ['id','publisher','jurisdiction','source_title','source_url']}|{'blocks':blocks,'sections':sections}

def main():
 p=argparse.ArgumentParser();p.add_argument('--documents',type=Path,required=True);p.add_argument('--local-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise ValueError('Preserve previous catalog; use a new snapshot name')
 counts=Counter();total=0
 with a.output.open('x',encoding='utf8') as f:
  for line in a.documents.read_text(encoding='utf8').splitlines():
   doc=json.loads(line);meta_path=a.local_root/doc['source_artifact'];meta=json.loads(meta_path.read_text(encoding='utf8'))
   row=catalog_document(doc,(meta_path.parent/meta['artifact']).read_bytes())
   for b in row['blocks']:assert doc['text'][b['start']:b['end']]==b['text'] and digest(b['text'])==b['sha256']
   f.write(json.dumps(row,ensure_ascii=False)+'\n');total+=1
   counts['blocks']+=len(row['blocks']);counts['sections']+=len(row['sections']);counts['tables']+=sum(b['element'] in ['table','va-table'] for b in row['blocks'])
 print(json.dumps({'documents':total,**counts,'accepted_items':0}))
if __name__=='__main__':main()
