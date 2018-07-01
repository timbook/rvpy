import unittest
import sys
import random

sys.path.append('..')

import rvpy

class NormalTests(unittest.TestCase):
    def setUp(self):
        self.rnd1 = random.random()
        self.rnd2 = random.random()
        self.Z = rvpy.StandardNormal()
        self.N1 = rvpy.Normal(mu=self.rnd1, sigma=self.rnd2)

    def test_normal_moments(self):
        self.assertEqual(self.Z.mean, 0)
        self.assertEqual(self.Z.std, 1)
        self.assertEqual(self.Z.var, 1)
        self.assertEqual(self.N1.mean, self.rnd1)
        self.assertEqual(self.N1.std, self.rnd2)
        self.assertEqual(self.N1.var, self.rnd2**2)

    def test_normal_pdf(self):
        self.assertAlmostEqual(self.Z.pdf(10), 0)
        self.assertAlmostEqual(self.Z.pdf(-10), 0)
        self.assertAlmostEqual(self.N1.pdf(10), 0)
        self.assertAlmostEqual(self.N1.pdf(-10), 0)

    def test_normal_cdf(self):
        self.assertAlmostEqual(self.Z.cdf(10), 1)
        self.assertAlmostEqual(self.Z.cdf(-10), 0)
        self.assertAlmostEqual(self.Z.cdf(0), 0.5)
        self.assertAlmostEqual(self.N1.cdf(10), 1)
        self.assertAlmostEqual(self.N1.cdf(-10), 0)
        self.assertAlmostEqual(self.N1.cdf(self.rnd1), 0.5)

    def test_normal_conversion(self):
        X = rvpy.Normal(0, 1)
        self.assertIsInstance(self.Z.to_nonstandard(), rvpy.Normal)
        self.assertIsInstance(X.to_standard(), rvpy.StandardNormal)
        with self.assertRaises(ValueError):
            self.N1.to_standard()

    def test_test_add(self):
        N1 = self.N1
        m = random.random()
        s = random.random()
        N2 = rvpy.Normal(mu=m, sigma=s)
        N3 = N1 + N2
        new_mu = N1.mu + m
        new_sigma = (N1.sigma**2 + s**2)**0.5
        self.assertAlmostEqual(N3.mean, new_mu)
        self.assertAlmostEqual(N3.sigma, new_sigma)
        



