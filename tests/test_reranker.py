import unittest
import importlib.util
import numpy as np

from src.insurerag_vlm.reranker import blend_scores, minmax_scores


class ScoreFusionTests(unittest.TestCase):
    def test_normalization_ignores_absolute_channel_scale(self):
        first = blend_scores([.5, .8, .6], [0, 12, 20], [4, -2, 8], .2, .75)
        second = blend_scores([5, 8, 6], [100, 220, 300], [18, 6, 26], .2, .75)
        np.testing.assert_allclose(first, second)

    def test_constant_channel_contributes_no_fake_discrimination(self):
        np.testing.assert_array_equal(minmax_scores([2, 2, 2]), [0, 0, 0])
        np.testing.assert_allclose(blend_scores([.1, .5], [5, 5]), [0, .8])

    def test_candidate_permutation_preserves_id_to_score(self):
        d = np.array([.1, .5, .7]); b = np.array([20, 5, 0]); c = np.array([3, 0, 9])
        perm = [2, 0, 1]
        np.testing.assert_allclose(blend_scores(d, b, c, .2, .75)[perm],
                                   blend_scores(d[perm], b[perm], c[perm], .2, .75))

    def test_zero_cross_weight_does_not_require_optional_model(self):
        np.testing.assert_allclose(blend_scores([.2, .8], [9, 0], lexical_weight=0), [0, 1])
        self.assertEqual(len(blend_scores([], [])), 0)

    def test_invalid_scores_weights_and_misaligned_candidates_fail(self):
        for kwargs in [{'lexical_weight': -1}, {'cross_weight': 2}, {'cross_weight': .5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                blend_scores([.2, .5], [1, 2], **kwargs)
        for values in [[float('nan')], [float('inf')], [[1, 2]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                minmax_scores(values)
        with self.assertRaises(ValueError):
            blend_scores([1, 2], [1])
        with self.assertRaises(ValueError):
            blend_scores([1, 2], [1, 2], [1], cross_weight=.5)


@unittest.skipUnless(importlib.util.find_spec('sklearn') and importlib.util.find_spec('scipy'),
                     'optional research evaluation dependencies')
class RerankCandidateContractTests(unittest.TestCase):
    def test_reranking_cannot_inject_a_gold_answer_outside_candidates(self):
        from scripts.eval_insuranceqa_reranker import rank_row
        row = {'candidate_ids': ['1', '2', '3'], 'dense': [.8, .6, .4],
               'sparse': [0, 5, 6], 'cross': [2, 1, 20], 'bge_ids': ['1', '2']}
        config = {'pool': 'bge100', 'lexical_weight': .2, 'cross_weight': 1}
        self.assertEqual(rank_row(row, config), ['1', '2'])
        config['pool'] = 'union200'
        self.assertEqual(rank_row(row, config), ['3', '1', '2'])

    def test_unknown_candidate_policy_rejected(self):
        from scripts.eval_insuranceqa_reranker import rank_row
        row = {'candidate_ids': [], 'dense': [], 'sparse': [], 'cross': []}
        with self.assertRaises(ValueError):
            rank_row(row, {'pool': 'oracle'})
