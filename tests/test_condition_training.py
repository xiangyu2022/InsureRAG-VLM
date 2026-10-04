"""Protect ambiguity handling and author-label isolation in the new training recipe."""
import random
import unittest
from scripts.train_condition_reranker import sample,confidence
from scripts.extract_additional_government import extract

class ConditionSamplingTests(unittest.TestCase):
    def group(self):
        sources={**{f's{i}':'student_hard' for i in range(6)},**{f'l{i}':'lexical_hard' for i in range(4)},'r':'random'}
        return {'positive_ids':['p1','p2'],'negative_ids':list(sources),'negative_sources':sources}
    def test_repeated_presentations_cover_author_positives_without_promoting_negatives(self):
        g=self.group();rng=random.Random(42);positives=set()
        for repeat in range(20):
            chosen=sample(g,rng,repeat);positives.add(chosen[0])
            self.assertEqual(len(set(chosen)),6)
            self.assertTrue(set(chosen[1:]).isdisjoint(g['positive_ids']))
        self.assertEqual(positives,set(g['positive_ids']))
    def test_ambiguous_negatives_have_nonzero_reduced_penalty(self):
        self.assertEqual(confidence([4,3,3.01,2.99,9,-3]),[.25,.25,1.,.25,1.])
    def test_teacher_logit_translation_does_not_change_confidence(self):
        scores=[5,3,4,5,6,-2]
        self.assertEqual(confidence(scores),confidence([s+20 for s in scores]))
    def test_sampler_refuses_missing_category_and_positive_negative_overlap(self):
        g=self.group();g['negative_ids'].remove('r')
        with self.assertRaises(ValueError):sample(g,random.Random(1),0)
        g=self.group();g['positive_ids']=['r']
        with self.assertRaises(ValueError):sample(g,random.Random(1),0)

class PublisherExtractionTests(unittest.TestCase):
    answer='The plan must evaluate the participant eligibility requirements and the applicable enrollment conditions before determining whether benefits are payable under the terms of this insurance coverage.'
    def test_mismatched_numbered_answer_is_rejected(self):
        rows,_=extract('Q1: What eligibility requirements apply to this coverage?\nA2: '+self.answer)
        self.assertEqual(rows,[])
    def test_wrapped_question_and_answer_preserve_original_spans(self):
        text='Q1: What eligibility requirements\napply to this coverage?\nA1: '+self.answer
        rows,_=extract(text);self.assertEqual(len(rows),1);row=rows[0]
        s,e=row['answer_span'];self.assertEqual(' '.join(text[s:e].split()),row['text'])
        s,e=row['question_span'];self.assertEqual(' '.join(text[s:e].split()),row['question'])
    def test_next_question_is_not_used_as_an_answer(self):
        text='Q1: What eligibility requirements apply to this coverage?\nA1: What are the rules for the following participants and their plans and how do the regulations apply to each relevant insurance case and eligibility status?'
        self.assertEqual(extract(text)[0],[])
    def test_short_uninformative_answer_is_rejected(self):
        self.assertEqual(extract('Q1: What eligibility requirements apply to this coverage?\nA1: Yes.')[0],[])
    def test_late_embedded_next_question_is_rejected(self):
        text='Frequently Asked Questions\n\nWhat requirements apply to eligibility for this coverage?\n\n'+self.answer+' 7. If the claimant has another plan, can the claimant still enroll? The employer must decide.'
        self.assertEqual(extract(text)[0],[])

if __name__=='__main__':unittest.main()
