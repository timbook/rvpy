import unittest
import sys
import random
import numpy as np

sys.path.append('..')

import rvpy

class NegativeBinomialTests(unittest.TestCase):
    def setUp(self):
        self.p = p = random.random()
        self.NB1 = rvpy.NegativeBinomial(random.randint(2, 11), p)
        self.NB2 = rvpy.NegativeBinomial(random.randint(2, 11), p)
        self.NB3 = rvpy.NegativeBinomial(1, p)
        self.G1 = rvpy.Geometric(p)
        self.G2 = rvpy.Geometric(p)

    def test_negbin_moments(self):
        self.assertAlmostEqual(self.G1.mean, self.G1.q / self.G1.p)
        self.assertAlmostEqual(self.G1.var, self.G1.q / self.G1.p**2)

        self.assertAlmostEqual(
                self.NB1.mean,
                self.NB1.r * (self.NB1.q / self.NB1.p)
        )
        self.assertAlmostEqual(
                self.NB1.var,
                self.NB1.r * (self.NB1.q / self.NB1.p**2)
        )

    def test_negbin_conversion(self):
        NBconv = self.NB3.to_geometric()
        self.assertIsInstance(NBconv, rvpy.Geometric)
        self.assertEqual(NBconv.mean, self.NB3.mean)
        self.assertEqual(NBconv.var, self.NB3.var)

        Gconv = self.G1.to_negative_binomial()
        self.assertNotIsInstance(Gconv, rvpy.Geometric)
        self.assertEqual(Gconv.mean, self.G1.mean)
        self.assertEqual(Gconv.var, self.G1.var)

    def test_negbin_pmf(self):
        self.assertEqual(self.NB1.pmf(-1), 0)
        self.assertEqual(self.G1.pmf(-1), 0)

    def test_negbin_add_sub(self):
        NB12 = self.NB1 + self.NB2
        self.assertIsInstance(NB12, rvpy.NegativeBinomial)
        self.assertEqual(NB12.p, self.p)
        self.assertEqual(NB12.r, self.NB1.r + self.NB2.r)

        NBG = self.NB1 + self.G1
        self.assertIsInstance(NBG, rvpy.NegativeBinomial)
        self.assertEqual(NBG.p, self.p)
        self.assertEqual(NBG.r, self.NB1.r + 1)

        Gplus = self.G1 + self.G2
        self.assertIsInstance(Gplus, rvpy.NegativeBinomial)
        self.assertEqual(Gplus.p, self.p)
        self.assertEqual(Gplus.r, 2)

    def test_negbin_errors(self):
        with self.assertRaises(AssertionError): rvpy.Geometric(0)
        with self.assertRaises(AssertionError): rvpy.Geometric(1)
        with self.assertRaises(AssertionError): rvpy.Geometric(-0.1)
        with self.assertRaises(AssertionError): rvpy.Geometric(1.1)

        with self.assertRaises(AssertionError): rvpy.NegativeBinomial(0, 0.1)
        with self.assertRaises(AssertionError): rvpy.NegativeBinomial(-1, 0.1)
        with self.assertRaises(AssertionError): rvpy.NegativeBinomial(0.5, 0.1)
        with self.assertRaises(AssertionError): rvpy.NegativeBinomial(3, -0.1)
        with self.assertRaises(AssertionError): rvpy.NegativeBinomial(3, 1.1)
