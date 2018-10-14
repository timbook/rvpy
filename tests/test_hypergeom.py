import unittest
import sys
import random

sys.path.append('..')

import rvpy

class HypergeometricTests(unittest.TestCase):
    def setUp(self):
        self.M = random.randint(2, 11)
        self.K = random.randint(2, 11)
        self.N = max(self.K, self.M) + random.randint(2, 11)

        self.X = rvpy.Hypergeometric(self.N, self.M, self.K)

    def test_hg_moments(self):

        def get_var(N, M, K):
            a = K * M / N
            b = (N - M) * (N - K)
            c = N * (N - 1)
            return a * b / c

        self.assertAlmostEqual(self.X.mean, self.K * self.M / self.N)
        self.assertAlmostEqual(self.X.var, get_var(self.N, self.M, self.K))

    def test_hg_pmf(self):
        self.assertEqual(self.X.pmf(-1), 0)
        self.assertEqual(self.X.pmf(self.X.N + 1), 0)
        self.assertEqual(self.X.pmf(0.5), 0)

    def test_hg_cdf(self):
        self.assertEqual(self.X.cdf(-1), 0)
        self.assertEqual(self.X.cdf(self.X.N + 1), 1)

    def test_hg_errors(self):
        with self.assertRaises(AssertionError): rvpy.Hypergeometric(10, 11, 5)
        with self.assertRaises(AssertionError): rvpy.Hypergeometric(10, 5, 11)
