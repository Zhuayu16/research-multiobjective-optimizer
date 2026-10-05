"""Independent known areas/fronts validate the benchmark reporting mathematics."""
import unittest
import numpy as np
from scripts.validate_feasibility import front2, hypervolume2, igd_mean, mixed_reference, mixed, quadratic, zdt1


class ValidationMetricTests(unittest.TestCase):
    def test_union_area_and_dominated_points(self):
        self.assertAlmostEqual(hypervolume2([[.2,.8],[.8,.2]],(1,1)),.28)
        self.assertAlmostEqual(hypervolume2([[.2,.8],[.8,.2],[.9,.9],[2,0]],(1,1)),.28)
        self.assertEqual(hypervolume2([[2,2]],(1,1)),0)

    def test_front_duplicates_and_ties(self):
        values=np.array([[0,1],[1,0],[0,2],[1,0],[1,1]])
        np.testing.assert_array_equal(values[front2(values)],[[0,1],[1,0]])

    def test_reference_distance_known_shift(self):
        self.assertEqual(igd_mean([[0,0],[1,1]],[[0,0],[1,1]]),0)
        self.assertEqual(igd_mean([[0,0]],[[3,4]]),5)

    def test_independent_optima_and_zdt_front(self):
        self.assertEqual(quadratic([[.37,-.23]])[0],0)
        reference=mixed_reference().iloc[0]
        np.testing.assert_allclose([reference.x,reference.k,reference.d],[.45,2,.9])
        self.assertAlmostEqual(reference.loss,.016)
        self.assertGreaterEqual(mixed([[reference.x,reference.k,reference.d]])[0,1],301.5)
        np.testing.assert_allclose(zdt1([[0,0],[.25,0],[1,0]]),[[0,1],[.25,.5],[1,0]])
