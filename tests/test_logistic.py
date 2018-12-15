import unittest
import sys
import random
from math import pi

sys.path.append('..')

import rvpy

class LogisticTests(unittest.TestCase):
    def setUp(self):
        self.loc = random.randint(-10, 10)
        self.scale = 10*random.random()
        self.X = rvpy.Logistic(self.loc, self.scale)

        self.a = 10*random.random()
        self.b = 10*random.random()

    def test_logistic_moments(self):
        self.assertEqual(self.X.mean, self.loc)
        self.assertAlmostEqual(self.X.var, self.scale**2 * pi**2 / 3)

    def test_logistic_cdf(self):
        self.assertEqual(self.X.cdf(self.loc), 0.5)

    def test_logistic_conversion(self):
        # TODO: For Gumbel, Log-Logistic
        pass

    def test_logistic_add_sub(self):
        Y1 = self.X + self.a
        Y2 = self.X - self.a
        self.assertEqual(Y1.mean, self.X.mean + self.a)
        self.assertEqual(Y2.mean, self.X.mean - self.a)
        self.assertEqual(Y1.var, self.X.var)
        self.assertEqual(Y2.var, self.X.var)

    def test_logistic_mul_div(self):
        Y1 = self.X * self.b
        Y2 = self.X / self.b

        self.assertEqual(Y1.loc, self.X.loc * self.b)
        self.assertAlmostEqual(Y2.loc, self.X.loc / self.b)

        self.assertEqual(Y1.scale, self.X.scale * self.b)
        self.assertAlmostEqual(Y2.scale, self.X.scale / self.b)

    def test_logistic_errors(self):
        with self.assertRaises(AssertionError): rvpy.Logistic(0, -1)
        with self.assertRaises(TypeError): rvpy.Logistic(0, 1) + rvpy.Logistic(0, 2)
