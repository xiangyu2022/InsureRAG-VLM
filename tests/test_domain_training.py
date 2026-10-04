import importlib.util
import random
import unittest

from scripts.train_domain_reranker import sample_group
from scripts.prepare_hicric_extension import extract_pairs, clean


class TrainingSampleTests(unittest.TestCase):
    def group(self):
        return {'positive_ids':['p1','p2'], 'negative_ids':['d1','d2','d3','d4','b1','b2','b3','b4','r1'],
                'negative_sources':{**{f'd{i}':'dense_hard' for i in range(1,5)},
                                    **{f'b{i}':'lexical_hard' for i in range(1,5)},'r1':'random'}}

    def test_positive_cycles_without_changing_negative_budget(self):
        for epoch, positive in [(1,'p1'),(2,'p2'),(3,'p1')]:
            chosen = sample_group(self.group(),random.Random(42),epoch)
            self.assertEqual(chosen[0],positive)
            self.assertEqual(len(set(chosen)),6)
            self.assertEqual(sum(x.startswith('d') for x in chosen),2)
            self.assertEqual(sum(x.startswith('b') for x in chosen),2)
            self.assertEqual(chosen[-1],'r1')

    def test_missing_hard_negatives_fail_instead_of_using_positive_as_negative(self):
        group=self.group(); group['negative_ids']=['d1','b1','b2','r1']
        with self.assertRaises(ValueError): sample_group(group,random.Random(42),1)

    def test_duplicate_negative_across_channels_fails(self):
        group=self.group(); group['negative_ids']=['d1','d1','b1','b2','r1']
        with self.assertRaises(ValueError): sample_group(group,random.Random(42),1)


class PublisherQuestionExtractionTests(unittest.TestCase):
    def test_spans_recover_exact_source_and_same_question_number(self):
        answer=' '.join(['coverage']*26)
        raw=f'Introduction\nQ12. How does the state determine eligibility?\nA12. {answer}\nQ13. How does a member request review?\nA13. {answer}'
        pairs,rejected=extract_pairs(raw)
        self.assertEqual(len(pairs),2); self.assertEqual(rejected,[])
        for pair in pairs:
            self.assertEqual(clean(raw[slice(*pair['question_span'])]),pair['question'])
            self.assertEqual(clean(raw[slice(*pair['answer_span'])]),pair['answer'])
            self.assertNotIn('Q13.',pair['answer'])

    def test_wrong_answer_number_cannot_be_labeled_positive(self):
        pairs,rejected=extract_pairs('Q1. How does the state determine eligibility?\nA2. '+'coverage '*30)
        self.assertEqual(pairs,[])
        self.assertEqual(rejected[0]['reason'],'missing_matching_answer_tag')

    def test_context_dependent_and_long_answers_excluded_before_inference(self):
        for question,answer,reason in [
            ('How does question 2 apply to coverage?','coverage '*30,'question_depends_on_omitted_context'),
            ('How does the state determine eligibility?','coverage '*351,'answer_length_outside_25_350_words')]:
            pairs,rejected=extract_pairs(f'Q1. {question}\nA1. {answer}')
            self.assertEqual(pairs,[]);self.assertEqual(rejected[0]['reason'],reason)


@unittest.skipUnless(importlib.util.find_spec('sklearn') and importlib.util.find_spec('scipy'),
                     'optional research evaluation dependencies')
class ResearchInferenceIntegrityTests(unittest.TestCase):
    def test_weight_or_tokenizer_change_is_rejected(self):
        from scripts.query_domain_reranker import verify_files
        verify_files({'model.safetensors':'a','tokenizer.json':'b','download_provenance.json':'old'},
                     {'model.safetensors':'a','tokenizer.json':'b','download_provenance.json':'new'})
        for modified in [{'model.safetensors':'changed','tokenizer.json':'b'},
                         {'model.safetensors':'a','tokenizer.json':'changed'}]:
            with self.assertRaises(ValueError):
                verify_files(modified,{'model.safetensors':'a','tokenizer.json':'b'})

    def test_evaluation_cannot_reorder_questions_against_labels(self):
        from scripts.eval_domain_training import predictions
        with self.assertRaises(ValueError):
            predictions([{'id':'a'}],[{'id':'b'}],{},'candidate')


if __name__=='__main__':
    unittest.main()
