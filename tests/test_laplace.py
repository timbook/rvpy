import unittest
import sys
import random

sys.path.append('..')

import rvpy

class LaplaceTests(unittest.TestCase):
    def setUp(self):
        b1 = random.expovariate(1/10)
        b2 = random.expovariate(1/10)
        mu = random.randint(-10, 10)

        self.X1 = rvpy.Laplace(0, b1)
        self.X2 = rvpy.Laplace(0, b1)
        self.Y = rvpy.Laplace(mu, b2)

    def test_laplace_moments(self):
        self.assertEqual(self.X1.mean, 0)
        self.assertEqual(self.X1.var, 2*self.X1.b**2)
        self.assertEqual(self.X1.skew, 0)
        self.assertEqual(self.X1.kurtosis, 3)

        self.assertEqual(self.Y.mean, self.Y.mu)
        self.assertEqual(self.Y.var, 2*self.Y.b**2)
        self.assertEqual(self.Y.skew, 0)
        self.assertEqual(self.Y.kurtosis, 3)

    def test_laplace_cdf(self):
        self.assertEqual(self.X1.cdf(0), 0.5)
        self.assertEqual(self.Y.cdf(self.Y.mu), 0.5)

    def test_laplace_conversion(self):
        Xabs = self.X1.abs()
        self.assertIsInstance(Xabs, rvpy.Exponential)
        self.assertEqual(Xabs.scale, self.X1.b)

    def test_laplace_add_sub(self):
        c = random.randint(1, 11)

        Yadd = self.Y + c
        self.assertEqual(Yadd.mu, self.Y.mu + c)
        self.assertEqual(Yadd.b, self.Y.b)

    def test_laplace_mul_div(self):
        c = random.expovariate(1/10)

        Ymul = c*self.Y
        self.assertAlmostEqual(Ymul.mu, c*self.Y.mu)
        self.assertAlmostEqual(Ymul.b, c*self.Y.b)

        Ydiv = self.Y / c
        self.assertAlmostEqual(Ydiv.mu, self.Y.mu / c)
        self.assertAlmostEqual(Ydiv.b, self.Y.b / c)


    def test_laplace_errors(self):
        with self.assertRaises(AssertionError): rvpy.Laplace(0, 0)
        with self.assertRaises(AssertionError): rvpy.Laplace(0, -1)
        with self.assertRaises(AssertionError):
            rvpy.Laplace(3, 3).abs()

