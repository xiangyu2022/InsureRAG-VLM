import random
import unittest
import numpy as np
try:
    from scripts.prepare_multidomain_data import canonical_url,extract_government
except ModuleNotFoundError as exc:
    if exc.name=='sklearn':raise unittest.SkipTest('Data preparation tests require optional scikit-learn') from exc
    raise
from scripts.train_domain_reranker import sample_group
from src.insurerag_vlm.retention import anchored_cross_scores
from scripts.eval_retention_v2 import predict,source_paired


class RetentionTests(unittest.TestCase):
    def test_wrong_number_does_not_become_a_label(self):
        text='Q1. Which rules apply to this example?\nA2. '+('An answer with enough words to pass the length requirement. '*4)
        pairs,rejected=extract_government(text)
        self.assertEqual(pairs,[]);self.assertEqual(rejected['answer_number_mismatch'],1)

    def test_paragraph_boundary_requires_publisher_faq(self):
        text='Q. Which rules apply to this example?\n\n'+('An answer with enough words to pass the length requirement. '*4)
        self.assertEqual(extract_government(text)[0],[])
        source='Frequently asked questions\n'+text
        pair=extract_government(source)[0][0]
        self.assertEqual(pair['question'],'Which rules apply to this example?')
        self.assertEqual(' '.join(source[slice(*pair['answer_span'])].split()),pair['text'])

    def test_archive_variants_share_source_group(self):
        self.assertEqual(canonical_url('https://web.archive.org/web/20200101/https://www.Medicaid.gov/media/123/'),
                         canonical_url('http://medicaid.gov/media/123'))

    def test_anchor_endpoints_and_invalid_inputs(self):
        np.testing.assert_allclose(anchored_cross_scores([1,2,3],[3,2,1],0),[1,.5,0])
        np.testing.assert_allclose(anchored_cross_scores([1,2,3],[3,2,1],1),[0,.5,1])
        np.testing.assert_allclose(anchored_cross_scores([1,2,3],[3,2,1],.5),[.5,.5,.5])
        with self.assertRaises(ValueError):anchored_cross_scores([1],[1,2])
        with self.assertRaises(ValueError):anchored_cross_scores([1],[2],1.01)
        with self.assertRaises(ValueError):anchored_cross_scores([float('nan')],[2])

    def test_prediction_rejects_candidate_mismatch(self):
        row={'id':'q','candidate_ids':['a'],'dense':[1.],'sparse':[1.],'bge_ids':['a'],'bm25_ids':['a'],'cross':[1.]}
        case={'id':'q','gold_answer_ids':['a']}
        self.assertEqual(predict([row],[row],[case],1.,'new')[0]['hit_at_10'],1.)
        with self.assertRaises(ValueError):predict([row],[{**row,'candidate_ids':['b']}],[case],.5,'new')

    def test_cluster_bootstrap_is_paired_and_source_based(self):
        cases=[{'source_group':'a'},{'source_group':'a'},{'source_group':'b'}]
        result=source_paired([{'hit_at_10':1}]*3,[{'hit_at_10':0}]*3,cases)
        self.assertEqual(result['source_groups'],2)
        self.assertEqual(result['source_cluster_bootstrap_95ci'],[1.,1.])

    def test_sampler_does_not_treat_teacher_as_gold(self):
        negatives=[str(i) for i in range(9)]
        group={'positive_ids':['p','p2'],'negative_ids':negatives,
               'negative_sources':{n:('dense_hard' if i<4 else 'lexical_hard' if i<8 else 'random') for i,n in enumerate(negatives)},
               'teacher_scores':{'p':-100,'0':100}}
        for epoch in [1,2]:
            picked=sample_group(group,random.Random(42),epoch)
            self.assertEqual(picked[0],group['positive_ids'][epoch-1]);self.assertEqual(len(set(picked)),6)


if __name__=='__main__':unittest.main()
