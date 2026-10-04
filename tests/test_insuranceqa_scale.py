import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.prepare_insuranceqa import parse_question,audit_splits,write_json,sha,verify_fixture


class ImportIntegrityTests(unittest.TestCase):
    def test_decode_does_not_add_gold_to_official_pool(self):
        case=parse_question('life\tidx_1 idx_2\t9\t1 2 2\n',{'idx_1':'What','idx_2':'coverage?'},'test',1)
        self.assertEqual(case['question'],'What coverage?')
        self.assertEqual(case['official_pool_ids'],['1','2'])
        self.assertNotIn('9',case['official_pool_ids'])

    def test_unknown_vocabulary_fails(self):
        with self.assertRaises(KeyError):parse_question('life\tidx_404\t1\t1',{},'test',1)

    def test_overlap_and_unknown_labels(self):
        rows=[parse_question('life\tWhat cost?\t1\t1',{},s,1) for s in ['train','test']]
        audit=audit_splits(rows,[{'id':'1','text':'Answer'}])
        self.assertEqual(audit['test_exact_overlap_with_train_or_valid'],1)
        rows[0]['gold_answer_ids']=['unknown']
        with self.assertRaises(ValueError):audit_splits(rows,[{'id':'1','text':'Answer'}])

    def test_lock_detects_change(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'test.jsonl').write_text('original')
            write_json({'files':{'test.jsonl':sha(root/'test.jsonl')}},root/'manifest.lock.json')
            verify_fixture(root)
            (root/'test.jsonl').write_text('changed')
            with self.assertRaises(ValueError):verify_fixture(root)


@unittest.skipUnless(importlib.util.find_spec('sklearn') and importlib.util.find_spec('scipy'),'optional scale benchmark dependencies')
class ScaleRetrievalTests(unittest.TestCase):
    def test_sparse_scores_match_existing_bm25(self):
        from scripts.eval_insuranceqa_scale import SparseBM25
        from src.insurerag_vlm.retriever import SparseRetriever
        texts=['flood flood policy deductible','health deductible coinsurance','unrelated text']
        fast=SparseBM25(texts)
        with tempfile.TemporaryDirectory() as d:
            reference=SparseRetriever();index=reference.build_index(texts,Path(d)/'index.json')
            actual=fast.scores('flood deductible deductible')
            for idx,score in reference.search('flood deductible deductible',index,top_k=3,return_scores=True):
                self.assertAlmostEqual(actual[idx],score,places=5)
            self.assertEqual(actual[2],0)

    def test_pool_missing_positive_stays_failure_and_ties_stable(self):
        from scripts.eval_insuranceqa_scale import ranked,metrics
        order=ranked(np.array([.9,.8,.8]),allowed=[1,2])
        self.assertEqual(order,[1,2])
        self.assertEqual(metrics(order,{0})['hit_at_100'],0)
        self.assertEqual(ranked(np.array([-.9,-.8])),[1,0])

    def test_multiple_gold_recall_is_not_hit_rate(self):
        from scripts.eval_insuranceqa_scale import metrics
        values=metrics([2,0,3],{0,1})
        self.assertEqual(values['hit_at_5'],1)
        self.assertEqual(values['label_recall_at_10'],.5)
        self.assertEqual(values['mrr_at_100'],.5)
