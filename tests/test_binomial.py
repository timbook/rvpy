import unittest
import sys
import random

sys.path.append('..')

import rvpy

class BinomialTests(unittest.TestCase):
    def setUp(self):
        self.px = random.random()
        self.nx = random.randrange(1, 10)
        self.X = rvpy.Binomial(self.nx, self.px)

        self.py = random.random()
        self.Y = rvpy.Bernoulli(self.py)

    def test_bin_moments(self):
        # Check moments of Binomials
        self.assertAlmostEqual(self.X.mean, self.X.p*self.X.n)
        self.assertAlmostEqual(self.X.var, self.X.q*self.X.p*self.X.n)
        self.assertAlmostEqual(self.X.std, self.X.var**0.5)

        # Check moments of Bernoulli
        self.assertAlmostEqual(self.Y.mean, self.Y.p)
        self.assertAlmostEqual(self.Y.var, self.Y.p*self.Y.q)
        self.assertAlmostEqual(self.Y.std, (self.Y.p*self.Y.q)**0.5)

    def test_bin_pmf(self):
        # Probability of support sums to 1
        self.assertAlmostEqual(self.Y.pmf(0) + self.Y.pmf(1), 1)
        self.assertAlmostEqual(sum(self.X.pmf(range(self.X.n + 1))), 1)

        # Probabilities outside support are 0
        self.assertEqual(self.X.pmf(-1), 0)
        self.assertEqual(self.X.pmf(self.X.n + 1), 0)
        self.assertEqual(self.Y.pmf(-1), 0)
        self.assertEqual(self.Y.pmf(2), 0)

    def test_bin_cdf(self):
        # Binomial CDF properties
        self.assertEqual(self.X.cdf(self.X.n), 1)
        self.assertEqual(self.X.cdf(0), (1 - self.X.p)**self.X.n)
        self.assertEqual(self.X.cdf(0), self.X.q**self.X.n)
        self.assertEqual(self.X.cdf(-1), 0)

        # Bernoulli CDF properties
        self.assertEqual(self.Y.cdf(0), 1 - self.Y.p)
        self.assertEqual(self.Y.cdf(0), self.Y.q)
        self.assertEqual(self.Y.cdf(1), 1)

    def test_bin_conversion(self):
        # TODO: This
        pass

    def test_bin_add(self):
        # TODO: This
        pass

    def test_bin_errors(self):
        # Parameter errors
        with self.assertRaises(AssertionError): rvpy.Binomial(-1, 0.1)
        with self.assertRaises(AssertionError): rvpy.Binomial(3, -0.1)
        with self.assertRaises(AssertionError): rvpy.Binomial(3, 0)
        with self.assertRaises(AssertionError): rvpy.Binomial(3, 1)
        with self.assertRaises(AssertionError): rvpy.Binomial(3, 1.1)

        with self.assertRaises(AssertionError): rvpy.Bernoulli(-0.1)
        with self.assertRaises(AssertionError): rvpy.Bernoulli(0)
        with self.assertRaises(AssertionError): rvpy.Bernoulli(1)
        with self.assertRaises(AssertionError): rvpy.Bernoulli(1.1)

        # Can't convert if n != 1
        with self.assertRaises(AssertionError): rvpy.Binomial(3, 0.5).to_bernoulli()
