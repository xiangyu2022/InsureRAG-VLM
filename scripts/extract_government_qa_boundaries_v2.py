"""Conservative explicit-Q/A parser for a future fixture, not condition_v1.

Recognize a question boundary only when a heading is followed by an actual
question and an explicit answer marker. A wrapped reference such as ``Q-A5.)``
must remain inside the answer. This post-test repair does not edit frozen data.
Implicit-answer and paragraph-only FAQs deliberately remain unsupported here.
"""
from collections import Counter
import re

from scripts.extract_additional_government import ANS, INTERROGATIVE, START, clean


HEADING = re.compile(
    r'(?mi)^[ \t]*(?:Question|Q(?:-[A-Z])?)[ \t]*(?P<number>\d{0,3})[.:)](?=[ \t\r\n])\s*'
)


def extract(text):
    raw = list(HEADING.finditer(text))
    validated = []
    rejected = Counter()
    for i, marker in enumerate(raw):
        end = raw[i + 1].start() if i + 1 < len(raw) else len(text)
        block = text[marker.end():end]
        answer = ANS.search(block)
        if answer is None:
            rejected['no_explicit_answer'] += 1
            continue
        qend = marker.end() + answer.start()
        question = clean(text[marker.end():qend])
        if not question.endswith('?') or not 5 <= len(question.split()) <= 120:
            rejected['not_a_question'] += 1
            continue
        qnumber = marker.group('number')
        anumber = re.findall(r'\d+', answer.group())
        if qnumber and anumber and qnumber != anumber[0]:
            rejected['number_mismatch'] += 1
            continue
        validated.append((marker, qend, marker.end() + answer.end(), question))
    pairs = []
    for i, (marker, qend, astart, question) in enumerate(validated):
        end = validated[i + 1][0].start() if i + 1 < len(validated) else len(text)
        answer = clean(text[astart:end])
        if not 25 <= len(answer.split()) <= 400:
            rejected['answer_length'] += 1
            continue
        if not START.search(answer) or INTERROGATIVE.search(answer) or '?' in answer:
            rejected['ambiguous_answer'] += 1
            continue
        # An unfinished reference is unsafe even if no false heading was found.
        if answer.count('(') != answer.count(')') or re.search(r'\b(?:and|or|see)\s*$', answer, re.I):
            rejected['unfinished_reference_or_clause'] += 1
            continue
        pairs.append({
            'question': question, 'text': answer,
            'question_span': [marker.end(), qend], 'answer_span': [astart, end],
            'extraction_method': 'validated_explicit_qa_boundaries_v2',
        })
    return pairs, dict(rejected)
