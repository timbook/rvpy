import unittest
import sys
import random 

import rvpy

sys.path.append('..')

class DUniformTests(unittest.TestCase):
    def setUp(self):
        self.X = rvpy.DUniform(0, 1)

        self.a = random.randint(2, 11)
        self.b = self.a + random.randint(2, 11)
        self.L = self.b - self.a + 1
        self.Y = rvpy.DUniform(self.a, self.b)

    def test_dunif_moments(self):
        self.assertEqual(self.X.mean, 0.5)
        self.assertEqual(self.X.var, 0.25)
        self.assertEqual(self.X.skew, 0)

        self.assertEqual(self.Y.mean, (self.b + self.a) / 2)
        self.assertEqual(self.Y.var, (self.L**2 - 1) / 12)
        self.assertEqual(self.Y.skew, 0)

    def test_dunif_pmf(self):
        self.assertEqual(self.X.pmf(0), 0.5)
        self.assertEqual(self.X.pmf(1), 0.5)
        self.assertEqual(self.X.pmf(0.5), 0)
        self.assertEqual(self.X.pmf(-1), 0)
        self.assertEqual(self.X.pmf(2), 0)

        self.assertEqual(self.Y.pmf(random.randint(self.a, self.b)), 1/self.L)
        self.assertEqual(self.Y.pmf(self.a - 1), 0)
        self.assertEqual(self.Y.pmf(self.b + 1), 0)
        self.assertEqual(self.Y.pmf(self.a + 0.5), 0)

    def test_dunif_cdf(self):
        self.assertEqual(self.X.cdf(-1), 0)
        self.assertEqual(self.X.cdf(0), 0.5)
        self.assertEqual(self.X.cdf(1), 1)
        self.assertEqual(self.X.cdf(2), 1)

        self.assertEqual(self.Y.cdf(self.a - 1), 0)
        self.assertEqual(self.Y.cdf(self.b + 1), 1)

    def test_dunif_add_sub(self):
        c = random.randint(1, 11)
        Yplus = self.Y + c
        self.assertEqual(Yplus.a, self.Y.a + c)
        self.assertEqual(Yplus.b, self.Y.b + c)
        self.assertEqual(Yplus.mean, self.Y.mean + c)
        self.assertEqual(Yplus.var, self.Y.var)

        Yminus = self.Y - c
        self.assertEqual(Yminus.a, self.Y.a - c)
        self.assertEqual(Yminus.b, self.Y.b - c)
        self.assertEqual(Yminus.mean, self.Y.mean - c)
        self.assertEqual(Yminus.var, self.Y.var)

    def test_dunif_errors(self):
        with self.assertRaises(AssertionError): rvpy.DUniform(0, 2.5)
        with self.assertRaises(AssertionError): rvpy.DUniform(2.5, 5)
        with self.assertRaises(AssertionError): rvpy.DUniform(3, 2)
