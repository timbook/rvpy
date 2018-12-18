import unittest
import sys
import random 
from math import exp, log

import rvpy

sys.path.append('..')

class GompertzTests(unittest.TestCase):
    def setUp(self):
        self.eta = 10*random.random()
        self.b = 10*random.random()
        self.X = rvpy.Gompertz(eta=self.eta, b=self.b)

    def test_gompertz_moments(self):
        self.assertAlmostEqual(
            self.X.median,
            1/self.b * log((-1/self.eta) * log(1/2) + 1)
        )
    
    def test_gompertz_pdf(self):
        x = 3 * random.random()
        self.assertAlmostEqual(
            self.X.pdf(x),
            self.b * self.eta * exp(self.b*x + self.eta) * exp(-self.eta * exp(self.b * x))
        )
        self.assertEqual(self.X.pdf(-3), 0)

    def test_gompertz_cdf(self):
        self.assertEqual(self.X.cdf(0), 0)

    def test_gompertz_errors(self):
        with self.assertRaises(AssertionError): rvpy.Gompertz(0, 1)
        with self.assertRaises(AssertionError): rvpy.Gompertz(1, 0)
        with self.assertRaises(AssertionError): rvpy.Gompertz(-1, 1)
        with self.assertRaises(AssertionError): rvpy.Gompertz(1, -1)
