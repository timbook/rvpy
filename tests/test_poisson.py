import unittest
import sys
import random
import numpy as np

sys.path.append('..')

import rvpy

class PoissonTests(unittest.TestCase):
    def setUp(self):
        self.mu1 = random.randrange(1, 11)
        self.mu2 = random.randrange(1, 11)

        self.X = rvpy.Poisson(self.mu1)
        self.Y = rvpy.Poisson(self.mu2)

    def test_poi_moments(self):
        self.assertEqual(self.X.mean, self.mu1)
        self.assertEqual(self.X.var, self.mu1)
        self.assertAlmostEqual(self.X.skew, 1/self.mu1**0.5)
        self.assertAlmostEqual(self.X.kurtosis, 1/self.mu1)

    def test_poi_pmf(self):
        self.assertAlmostEqual(self.X.pmf(0), np.exp(-self.X.mu))
        self.assertAlmostEqual(self.X.pmf(1), self.X.mu*np.exp(-self.X.mu))

    def test_poi_add_sub(self):
        Z = self.X + self.Y
        self.assertIsInstance(Z, rvpy.Poisson)
        self.assertEqual(Z.mean, self.X.mean + self.Y.mean)

    def test_poi_errors(self):
        with self.assertRaises(AssertionError): rvpy.Poisson(0)
        with self.assertRaises(AssertionError): rvpy.Poisson(-1)
