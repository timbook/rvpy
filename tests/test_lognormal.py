import unittest
import sys
import random 
import warnings
import numpy as np

import rvpy

sys.path.append('..')

class LogNormalTests(unittest.TestCase):
    def setUp(self):
        self.X = rvpy.LogNormal()
        self.Y = rvpy.LogNormal(-10 + 20*random.random(), 10*random.random())

    def test_lognormal_moments(self):
        self.assertAlmostEqual(self.X.mean, np.exp(1/2))
        self.assertAlmostEqual(
                np.log(self.X.var),
                np.log(np.exp(1) - 1) + 1
        )

        m = self.Y.mu
        v = self.Y.sigma**2
        self.assertAlmostEqual(np.log(self.Y.mean), m + 0.5*v)
        self.assertAlmostEqual(
                np.log(self.Y.var),
                np.log((np.exp(v) - 1)) + (2*m + v)
        )

    def test_lognormal_pdf(self):
        self.assertEqual(self.X.pdf(-1), 0)
        self.assertEqual(self.Y.pdf(-1), 0)

    def test_lognormal_cdf(self):
        self.assertEqual(self.X.cdf(-1), 0)
        self.assertEqual(self.Y.cdf(-1), 0)

    def test_lognormal_conversion(self):
        Xlog = self.X.log()
        Ylog = self.Y.log()

        self.assertIsInstance(Xlog, rvpy.Normal)
        self.assertIsInstance(Xlog.to_standard(), rvpy.StandardNormal)
        self.assertEqual(Xlog.mean, 0)
        self.assertEqual(Xlog.var, 1)

        self.assertIsInstance(Ylog, rvpy.Normal)
        self.assertEqual(Ylog.mean, self.Y.mu)
        self.assertEqual(Ylog.var, self.Y.var)

    def test_lognormal_mul_div_pow(self):
        a = random.expovariate(1/10)

        # Multiplication by constant
        Ymul = self.Y*a
        self.assertEqual(Ymul.mu, self.Y.mu + np.log(a))
        self.assertEqual(np.log(Ymul.sigma), np.log(self.Y.sigma))

        # Division by Constant
        Ydiv = self.Y / a
        self.assertEqual(Ydiv.mu, self.Y.mu - np.log(a))
        self.assertEqual(np.log(Ydiv.sigma), np.log(self.Y.sigma))

        # Multiplication by another LogNormal
        Y2 = rvpy.LogNormal(-10 + 20*random.random(), 10*random.random())
        Z = self.Y * Y2
        self.assertIsInstance(Z, rvpy.LogNormal)
        self.assertEqual(Z.mu, self.Y.mu + Y2.mu)
        self.assertEqual(Z.sigma, self.Y.sigma + Y2.sigma)

        # Inversion
        Yinv = 1 / self.Y
        self.assertIsInstance(Yinv, rvpy.LogNormal)
        self.assertEqual(Yinv.mu, -self.Y.mu)
        self.assertEqual(Yinv.sigma, self.Y.sigma)

    def test_lognormal_conversion(self):
        with self.assertRaises(AssertionError): rvpy.LogNormal(0, 0)
        with self.assertRaises(AssertionError): rvpy.LogNormal(0, -1)
        with self.assertRaises(TypeError): rvpy.LogNormal() + rvpy.LogNormal()
        with self.assertRaises(TypeError): rvpy.LogNormal()**0
        with self.assertRaises(TypeError): rvpy.LogNormal()*0


        

