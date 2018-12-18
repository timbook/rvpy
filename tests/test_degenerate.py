import unittest
import sys
import random 

import numpy as np

import rvpy

sys.path.append('..')

class DegenerateTests(unittest.TestCase):
    def setUp(self):
        self.k = random.randint(-10, 10)
        self.X = rvpy.Degenerate(self.k)

    def test_degenerate_moments(self):
        self.assertEqual(self.X.mean, self.k)
        self.assertEqual(self.X.var, 0)
        self.assertTrue(np.isnan(self.X.skew))
        self.assertTrue(np.isnan(self.X.kurtosis))

    def test_degenerate_pdf(self):
        self.assertEqual(self.X.pdf(self.k), 1)
        self.assertEqual(self.X.pdf(self.k + 0.0001), 0)

    def test_degenerate_cdf(self):
        self.assertEqual(self.X.cdf(self.k + 0.0001), 1)
        self.assertEqual(self.X.cdf(self.k - 0.0001), 0)
        self.assertEqual(self.X.cdf(self.k), 1)

    def test_degenerate_add_sub(self):
        c = 5*random.random()
        Y = self.X + c

        self.assertIsInstance(Y, rvpy.Degenerate)
        self.assertEqual(Y.mean, self.X.mean + c)
        self.assertEqual(Y.k, self.k + c)
