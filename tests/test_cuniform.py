import unittest
import sys
import random

sys.path.append('..')

import rvpy

class CUniformTests(unittest.TestCase):
    def setUp(self):
        # Standard uniform
        self.U = rvpy.CUniform()

        # Random nonstandard uniform
        self.a = random.random()
        self.b = self.a + 10*random.random()
        self.V = rvpy.CUniform(self.a, self.b)

    def test_cunif_moments(self):
        # Standard uniform moments
        self.assertEqual(self.U.mean, 0.5)
        self.assertEqual(self.U.median, 0.5)
        self.assertEqual(self.U.var, 1/12)
        self.assertEqual(self.U.skew, 0)
        self.assertEqual(self.U.kurtosis, -6/5)

        # Nonstandard uniform moments
        self.assertAlmostEqual(self.V.mean, (self.a + self.b) / 2)
        self.assertAlmostEqual(self.V.median, (self.a + self.b) / 2)
        self.assertAlmostEqual(self.V.var, (self.b - self.a)**2 / 12)
        self.assertEqual(self.V.skew, 0)
        self.assertEqual(self.V.kurtosis, -6/5)

    def test_cunif_pdf(self):
        # Standard uniform should be a block from 0 to 1
        self.assertEqual(self.U.pdf(0 - random.random()), 0)
        self.assertEqual(self.U.pdf(0 + random.random()), 1)
        self.assertEqual(self.U.pdf(random.random()), 1)
        self.assertEqual(self.U.pdf(1/2), 1)
        self.assertEqual(self.U.pdf(1 - random.random()), 1)
        self.assertEqual(self.U.pdf(1 + random.random()), 0)

        # Nonstandard uniform is a block from [a, b] of height 1 / (b - a)
        a = self.V.a
        b = self.V.b
        rnd = random.random()
        V_height = 1 / (b - a)

        self.assertEqual(self.V.pdf(a - rnd), 0)
        self.assertEqual(self.V.pdf(b*rnd + a), V_height)
        self.assertEqual(self.V.pdf((a + b) / 2), V_height)
        self.assertEqual(self.V.pdf(b - rnd), V_height)
        self.assertEqual(self.V.pdf(b + rnd), 0)

    def test_cunif_cdf(self):
        print("Unresolved TODOs!")

    def test_cunif_conversion(self):
        # TODO: Conversion to Beta 
        print("Unresolved TODOs!")

    def test_cunif_add_sub(self):
        print("Unresolved TODOs!")

    def test_cunif_mul_div(self):
        print("Unresolved TODOs!")
