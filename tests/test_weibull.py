import unittest
import sys
import random
from scipy.special import gamma as G

sys.path.append('..')

import rvpy

class WeibullTests(unittest.TestCase):
    def setUp(self):
        self.X = rvpy.Weibull(1, random.expovariate(1/10))
        self.Y = rvpy.Weibull(2, random.expovariate(1/10))
        self.Z = rvpy.Weibull(random.expovariate(1/10),
                              random.expovariate(1/10))

    def test_weibull_moments(self):
        self.assertEqual(self.X.mean, self.X.beta)
        self.assertAlmostEqual(self.X.var, self.X.beta**2 * (G(3) - 1))

        self.assertEqual(self.Y.mean, self.Y.beta * G(1.5))
        self.assertAlmostEqual(self.Y.var, self.Y.beta**2 * (G(2) - G(1.5)**2))

    def test_weibull_conversion(self):
        Xexp = self.X.to_exponential()
        self.assertIsInstance(Xexp, rvpy.Exponential)
        self.assertEqual(Xexp.mean, self.X.mean)
        self.assertEqual(Xexp.var, self.X.var)

        Yray = self.Y.to_rayleigh()
        self.assertIsInstance(Yray, rvpy.Rayleigh)
        self.assertEqual(Yray.mean, self.Y.mean)
        self.assertEqual(Yray.var, self.Y.var)

    def test_weibull_errors(self):
        with self.assertRaises(AssertionError): rvpy.Weibull(0, 3)
        with self.assertRaises(AssertionError): rvpy.Weibull(3, 0)
        with self.assertRaises(AssertionError): rvpy.Weibull(-1, 3)
        with self.assertRaises(AssertionError): rvpy.Weibull(3, -1)

class RayleighTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_rayleigh_moments(self):
        pass

    def test_rayleigh_conversion(self):
        pass

    def test_rayleigh_errors(self):
        pass
