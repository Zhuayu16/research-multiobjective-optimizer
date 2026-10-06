import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import pandas as pd
from mtpv_optimizer.core import _candidate_models, decision_scores, fit_surrogates, ideal_distances
from mtpv_optimizer.general import run_general
from mtpv_optimizer.presets import example
from mtpv_optimizer.scientific import correlations, decision_matrix, spatial_field
from mtpv_optimizer.nsga2 import optimize_nsga2


class ScientificEvidenceTests(unittest.TestCase):
    def test_attached_formula_differs_from_variance_weighting(self):
        y = np.array([[2., 8., 6.], [4., 5., 9.]])
        weights = np.array([1., 2., 3.]) / 6
        anchors = (np.zeros(3), np.full(3, 10.))
        loss = np.column_stack([y[:, 0]/10, 1-y[:, 1]/10, 1-y[:, 2]/10])
        expected = np.linalg.norm(weights * loss, axis=1)
        actual = ideal_distances(y, weights, anchors, ['min','max','max'], 'coefficient')
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.allclose(actual, ideal_distances(y, weights, anchors, ['min','max','max'])))
        frame = pd.DataFrame(y, columns=['a','b','c'])
        frame['综合效用'] = 1/(1+expected)
        audit = decision_matrix(frame, ['a','b','c'], ['min','max','max'], weights, anchors, 'IDEAL-COEFFICIENT')
        np.testing.assert_allclose(audit.D, expected)
        np.testing.assert_allclose(audit['效用复算差'], 0)
        np.testing.assert_allclose(audit.filter(like='加权平方::').sum(axis=1), audit.D**2)

    def test_spatial_interpolation_reproduces_a_known_plane_and_masks_outside(self):
        data = pd.DataFrame({'x':[0,1,0,.3], 'y':[0,0,1,.2]})
        data['T_K'] = 300+20*data.x-15*data.y
        result = spatial_field(data, 'x','y','T_K',16)
        finite = result['grid'].dropna()
        np.testing.assert_allclose(finite.T_K, 300+20*finite.x-15*finite.y, atol=1e-10)
        self.assertGreater(result['masked_cells'], 0)
        self.assertEqual(result['valid_cells']+result['masked_cells'],256)
        with self.assertRaisesRegex(ValueError,'同一位置'):
            spatial_field(pd.concat([data,pd.DataFrame({'x':[0],'y':[0],'T_K':[999]})]),'x','y','T_K')
        with self.assertRaisesRegex(ValueError,'不共线'):
            spatial_field(pd.DataFrame({'x':[0,1,2],'y':[0,0,0],'T':[1,2,3]}),'x','y','T')

    def test_correlations_preserve_unknown_constants_and_pair_counts(self):
        frame = pd.DataFrame({'x':[1,2,3,4], 'inverse':[-1,-2,-3,-4], 'constant':[5]*4})
        c = correlations(frame,list(frame))
        self.assertAlmostEqual(c['pearson'].loc['x','inverse'],-1)
        self.assertAlmostEqual(c['spearman'].loc['x','inverse'],-1)
        self.assertTrue(np.isnan(c['pearson'].loc['constant','constant']))
        self.assertEqual(c['pairs'].loc['x','inverse'],4)

    def test_new_models_actually_fit_and_predict_finite_held_out_values(self):
        rng = np.random.default_rng(17)
        frame = pd.DataFrame(rng.uniform(0,1,(30,2)),columns=['x','y'])
        frame['target'] = 2+frame.x-3*frame.y+.2*frame.x**3
        for name in ['Ridge-L2','ElasticNet','RSM-Cubic','KNN-Distance','AdaBoost']:
            with self.subTest(model=name):
                bundle = fit_surrogates(frame,['x','y'],['target'],model_mode=name)
                self.assertEqual(bundle.metrics[0].model_name,name)
                self.assertTrue(np.isfinite(bundle.validation_frame['交叉验证预测值']).all())
                self.assertEqual(bundle.training_audit['fits_completed'],6)
        self.assertEqual(len(_candidate_models(42,2)),13)

    def test_search_counters_match_observed_prediction_calls(self):
        problem, frame = example('battery')
        output = run_general(frame, replace(problem,population=20,generations=10,runs=1,decision='IDEAL-COEFFICIENT'))
        bundle=output.bundle
        sizes=[]
        original=bundle.predict
        def counted(x):
            sizes.append(len(np.atleast_2d(x)))
            return original(x)
        with patch.object(bundle,'predict',side_effect=counted):
            result=optimize_nsga2(bundle,20,10,42,problem.weights,2,'IDEAL-COEFFICIENT',anchors=(np.array(output.config['anchor_minimum']),np.array(output.config['anchor_maximum'])))
        self.assertEqual(result.evaluation_audit['objective_vector_rows'],sum(sizes))
        self.assertEqual(result.evaluation_audit['objective_prediction_batches'],len(sizes))
        self.assertEqual(len(result.history),20)
        self.assertEqual({r['generation'] for r in result.history},set(range(1,11)))
        self.assertLess(np.abs(output.decision_matrix['效用复算差']).max(),1e-12)
        self.assertTrue(all(0<=r['feasible_count']<=20 for r in result.history))

if __name__=='__main__':
    unittest.main()
