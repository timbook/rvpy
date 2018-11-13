import unittest
import sys
import random
from math import isnan

sys.path.append('..')

import rvpy

class CauchyTests(unittest.TestCase):
    def setUp(self):
        self.X = rvpy.StandardCauchy()
        self.Y = rvpy.Cauchy(random.randint(1, 11), random.expovariate(1/10))

    def test_cauchy_moments(self):
        self.assertTrue(isnan(self.X.mean))
        self.assertTrue(isnan(self.X.var))
        self.assertTrue(isnan(self.X.skew))
        self.assertTrue(isnan(self.X.kurtosis))
        self.assertEqual(self.X.median, 0)

        self.assertTrue(isnan(self.Y.mean))
        self.assertTrue(isnan(self.Y.var))
        self.assertTrue(isnan(self.Y.skew))
        self.assertTrue(isnan(self.Y.kurtosis))
        self.assertEqual(self.Y.median, self.Y.loc)
    
    def test_cauchy_cdf(self):
        self.assertAlmostEqual(self.X.cdf(0), 0.5)
        self.assertAlmostEqual(self.Y.cdf(self.Y.loc), 0.5)

    def test_cauchy_conversion(self):
        Z1 = rvpy.Cauchy().to_standard()
        self.assertIsInstance(Z1, rvpy.StandardCauchy)

        Z2 = self.X.to_nonstandard()
        self.assertIsInstance(Z2, rvpy.Cauchy)
        self.assertNotIsInstance(Z2, rvpy.StandardCauchy)

        Z3 = self.X.to_t()
        self.assertIsInstance(Z3, rvpy.T)
        self.assertEqual(Z3.df, 1)

    def test_cauchy_add_sub(self):
        loc1 = random.randint(1, 11)
        sc1 = random.expovariate(1/10)
        loc2 = random.randint(1, 11)
        sc2 = random.expovariate(1/10)

        a = random.expovariate(1/10)

        A = rvpy.Cauchy(loc1, sc1)
        B = rvpy.Cauchy(loc2, sc2)

        Cplus = A + B
        self.assertIsInstance(Cplus, rvpy.Cauchy)
        self.assertEqual(Cplus.loc, loc1 + loc2)
        self.assertEqual(Cplus.scale, sc1 + sc2)

        Cminus = A - B
        self.assertIsInstance(Cminus, rvpy.Cauchy)
        self.assertEqual(Cminus.loc, loc1 - loc2)
        self.assertEqual(Cminus.scale, sc1 + sc2)

        Aplusa = A + a
        self.assertIsInstance(Aplusa, rvpy.Cauchy)
        self.assertEqual(Aplusa.loc, loc1 + a)
        self.assertEqual(Aplusa.scale, sc1)

        Aminusa = A - a
        self.assertIsInstance(Aminusa, rvpy.Cauchy)
        self.assertEqual(Aminusa.loc, loc1 - a)
        self.assertEqual(Aminusa.scale, sc1)

    def test_cauchy_mul_div(self):
        c = random.randint(-10, 10)
        c = c if c != 0 else 1

        Ymul = c * self.Y
        self.assertIsInstance(Ymul, rvpy.Cauchy)
        self.assertEqual(Ymul.loc, c*self.Y.loc)
        self.assertEqual(Ymul.scale, abs(c)*self.Y.scale)

        Ydiv = self.Y / c
        self.assertIsInstance(Ydiv, rvpy.Cauchy)
        self.assertAlmostEqual(Ydiv.loc, self.Y.loc / c)
        self.assertAlmostEqual(Ydiv.scale, self.Y.scale / abs(c))

        d = random.randint(1, 10)
        Zinv = 1 / rvpy.Cauchy(0, d)
        self.assertIsInstance(Zinv, rvpy.Cauchy)
        self.assertEqual(Zinv.loc, 0)
        self.assertAlmostEqual(Zinv.scale, 1/d)

    def test_cauchy_errors(self):
        with self.assertRaises(AssertionError): rvpy.Cauchy(0, -1)
        with self.assertRaises(AssertionError):
            rvpy.Cauchy(3, 5).to_standard()
