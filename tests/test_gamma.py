import unittest
import sys
import random

sys.path.append('..')

import rvpy

class GammaTests(unittest.TestCase):
    def setUp(self):
        self.alpha1 = random.expovariate(1/10)
        self.alpha2 = random.expovariate(1/10)
        self.beta = random.expovariate(1/10)

        self.X = rvpy.Gamma(self.alpha1, self.beta)
        self.Y = rvpy.Gamma(self.alpha2, self.beta)

    def test_gamma_moments(self):
        self.assertAlmostEqual(self.X.mean, self.alpha1*self.beta)
        self.assertAlmostEqual(self.X.var, self.alpha1*self.beta**2)
        self.assertAlmostEqual(self.X.skew, 2/self.alpha1**0.5)
        self.assertAlmostEqual(self.X.kurtosis, 6/self.alpha1)

    def test_gamma_conversion(self):
        lambd = random.expovariate(1/5)
        U = rvpy.Gamma(1, lambd)
        self.assertIsInstance(U.to_exponential(), rvpy.Exponential)
        self.assertEqual(U.to_exponential().mean, lambd)

        alph = 2*random.randrange(3, 11)
        V = rvpy.Gamma(alph, 2)
        self.assertIsInstance(V.to_chisq(), rvpy.ChiSq)
        self.assertEqual(V.to_chisq().df, 2*alph)


    def test_gamma_add_sub(self):
        Z = self.X + self.Y
        self.assertEqual(Z.mean, (self.alpha1 + self.alpha2)*self.beta)
        self.assertEqual(Z.alpha, self.alpha1 + self.alpha2)
        self.assertEqual(Z.beta, self.beta)

    def test_gamma_mul_div(self):
        c = random.expovariate(1/10)

        Z1 = c*self.X
        Z2 = self.X*c
        Z3 = self.X / c

        self.assertAlmostEqual(Z1.beta, c*self.X.beta)
        self.assertAlmostEqual(Z2.beta, c*self.X.beta)
        self.assertAlmostEqual(Z3.beta, self.X.beta / c)

    def test_gamma_errors(self):
        with self.assertRaises(AssertionError): rvpy.Gamma(0, 1)
        with self.assertRaises(AssertionError): rvpy.Gamma(1, 0)
        with self.assertRaises(AssertionError): rvpy.Gamma(-1, 1)
        with self.assertRaises(AssertionError): rvpy.Gamma(1, -1)

class ExponentialTests(unittest.TestCase):
    pass

class ChiSqTests(unittest.TestCase):
    pass
