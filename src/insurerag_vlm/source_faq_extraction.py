"""Extract original FAQ structures; all output requires downstream QA review."""
import re
from bs4 import BeautifulSoup,Tag,NavigableString

def clean(text):return ' '.join(text.split())
def content_root(soup):
    main=soup.find('main') or soup.select_one('#main-content') or soup.select_one('#main') or soup
    for tag in main.find_all(['script','style','nav','header','footer']):tag.decompose()
    return main

def extract_original_faq(raw):
    soup=BeautifulSoup(raw,'html.parser');main=content_root(soup);headings=[]
    for node in main.find_all(['h1','h2','h3','h4','h5','h6','summary','strong','b','dt','button']):
        text=clean(node.get_text(' ',strip=True))
        if not text.endswith('?') or not 12<=len(text)<=450:continue
        if node.find_parent(['h1','h2','h3','h4','h5','h6','summary','strong','b','button','a']):continue
        if {'nav-link','tab-button'}&set(node.get('class',[])):continue
        if re.search(r'(more questions|was this|find what|helpful|help us|how can we help)',text,re.I):continue
        if node.name=='button' and not (node.get('aria-controls') or 'acc' in ' '.join(node.get('class',[]))):continue
        headings.append(node)
    identities={id(n) for n in headings};output=[]
    for node in headings:
        question=clean(node.get_text(' ',strip=True));answer=None;method='adjacent_dom';flags=[]
        control=node if node.name=='button' else node.find('button',attrs={'aria-controls':True})
        if control is not None and control.get('aria-controls'):
            panel=main.find(id=control['aria-controls'])
            if panel is not None:answer=clean(panel.get_text(' ',strip=True));method='explicit_accordion_panel'
        if answer is None and node.name=='button':
            # Theme accordions keep exactly one question and body in one item.
            parent=node.parent
            while parent is not None and parent is not main:
                classes=' '.join(parent.get('class',[]))
                if re.search(r'(?:^|\s)x-acc-item(?:\s|$)',classes):
                    body=parent.select_one('.x-acc-content')
                    if body is not None:answer=clean(body.get_text(' ',strip=True));method='theme_accordion_panel'
                    break
                parent=parent.parent
        if answer is None and node.name=='dt':
            sibling=node.find_next_sibling()
            if sibling is not None and sibling.name=='dd':answer=clean(sibling.get_text(' ',strip=True));method='definition_pair'
            else:continue
        if answer is None:
            parts=[]
            for nxt in node.next_elements:
                if node in nxt.parents:continue
                if main not in nxt.parents:break
                if isinstance(nxt,Tag):
                    if id(nxt) in identities:break
                    if nxt.name in ['h1','h2','h3','h4','h5','h6','summary']:
                        if node.name.startswith('h') and node.name[1:].isdigit() and nxt.name.startswith('h') and int(nxt.name[1:])>int(node.name[1:]):
                            flags.append('answer_contains_subsections')
                        else:break
                elif isinstance(nxt,NavigableString) and not nxt.find_parent(['script','style','button']):
                    text=clean(str(nxt))
                    if text:parts.append(text)
            answer=clean(' '.join(parts))
        if len(answer.split())<8 or len(answer)>25000:continue
        output.append({'question':question,'answer':answer,'extraction_method':method,'extraction_flags':sorted(set(flags))})
    return output
