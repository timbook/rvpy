import unittest
import sys
import random 

import rvpy

sys.path.append('..')

class BetaTests(unittest.TestCase):
    def setUp(self):
        # Beta rv with random params
        self.a = random.randint(10, 21)
        self.b = random.randint(10, 21)
        self.X = rvpy.Beta(self.a, self.b)

        # "Standard" beta
        self.Y = rvpy.Beta(1, 1)

    def test_beta_moments(self):
        X_exp_mean = self.a / (self.a + self.b)
        self.assertAlmostEqual(self.X.mean, X_exp_mean)

        X_exp_var = self.a*self.b / ((self.a + self.b)**2 * (self.a + self.b + 1))
        self.assertAlmostEqual(self.X.var, X_exp_var)

        self.assertAlmostEqual(self.Y.mean, 1/2)
        self.assertAlmostEqual(self.Y.var, 1/12)

    def test_beta_pdf(self):
        self.assertEqual(self.X.pdf(-0.1), 0)
        self.assertEqual(self.X.pdf(1.1), 0)

        self.assertEqual(self.Y.pdf(-0.1), 0)
        self.assertEqual(self.Y.pdf(1.1), 0)
        self.assertEqual(self.Y.pdf(1/2), 1)

    def test_beta_cdf(self):
        self.assertEqual(self.X.cdf(-0.1), 0)
        self.assertEqual(self.X.cdf(1.1), 1)

        self.assertEqual(self.Y.cdf(-0.1), 0)
        self.assertEqual(self.Y.cdf(1.1), 1)

        r = random.random()
        self.assertEqual(self.Y.cdf(r), r)

    def test_beta_conversion(self):
        self.assertIsInstance(self.Y.to_cuniform(), rvpy.CUniform)

    def test_beta_errors(self):
        with self.assertRaises(AssertionError): rvpy.Beta(0, 1)
        with self.assertRaises(AssertionError): rvpy.Beta(1, 0)
        with self.assertRaises(AssertionError): rvpy.Beta(-1, 1)
        with self.assertRaises(AssertionError): rvpy.Beta(1, -1)
