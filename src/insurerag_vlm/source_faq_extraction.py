"""Extract original FAQ structures; all output requires downstream QA review."""
import re
from bs4 import BeautifulSoup,Tag,NavigableString,Comment,ProcessingInstruction

from .source_mathml import preserve_mathml


def clean(text):return ' '.join(text.split())
def content_root(soup, *, preserve_headers=()):
    main=(soup.select_one('#main_content .content_left_column') or soup.find('main')
          or soup.select_one('[role="main"]') or soup.select_one('#main-main-content')
          or soup.select_one('#main-content') or soup.select_one('#main') or soup)
    # Some publishers use header for substantive section labels. Preserve only
    # explicitly reviewed selectors; navigation/header removal remains default.
    if not isinstance(preserve_headers, (list, tuple)):
        raise ValueError('Reviewed header selectors must be a list or tuple')
    selected = {}
    for selector in preserve_headers:
        if not isinstance(selector, str) or not selector.strip():
            raise ValueError('Reviewed header selector must be nonempty text')
        matches = main.select(selector)
        if not matches or any(node.name != 'header' for node in matches):
            raise ValueError('Reviewed header selector must match header elements')
        selected.update((id(node), node) for node in matches)
    removable = ['script', 'style', 'nav', 'header', 'footer']
    for node in selected.values():
        for parent in node.parents:
            if parent is main:
                break
            if parent.name in removable and id(parent) not in selected:
                raise ValueError('Reviewed header is inside excluded navigation')
    for comment in main.find_all(string=lambda value:isinstance(value,(Comment,ProcessingInstruction))):comment.extract()
    for tag in list(main.find_all(removable)):
        if tag.name is not None and id(tag) not in selected:
            tag.decompose()
    for link in main.find_all('a'):
        if clean(link.get_text(' ',strip=True)).casefold()=='shareable link to answer':link.decompose()
    preserve_mathml(main)
    return main

def extract_original_faq(raw, *, preserve_headers=()):
    soup=BeautifulSoup(raw,'html.parser');main=content_root(soup, preserve_headers=preserve_headers);headings=[]
    def question_text(node):
        return clean(node.get('header','') if node.name=='va-accordion-item' else node.get_text(' ',strip=True))
    for node in main.find_all(['h1','h2','h3','h4','h5','h6','summary','strong','b','dt','button','va-accordion-item']):
        text=question_text(node)
        if not text.endswith('?') or not 12<=len(text)<=450:continue
        if node.find_parent(['h1','h2','h3','h4','h5','h6','summary','strong','b','button','a','va-accordion-item']):continue
        if {'nav-link','tab-button'}&set(node.get('class',[])):continue
        if re.search(r'(more questions|was this|find what|helpful|help us|how can we help)',text,re.I):continue
        if node.name=='button' and not (node.get('aria-controls') or 'acc' in ' '.join(node.get('class',[]))):continue
        headings.append(node)
    identities={id(n) for n in headings};output=[]
    for node in headings:
        question=question_text(node);answer=None;method='adjacent_dom';flags=[]
        if node.name=='va-accordion-item':
            answer=clean(node.get_text(' ',strip=True));method='va_accordion_header_attribute'
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
