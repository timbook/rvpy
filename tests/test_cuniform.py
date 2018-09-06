import unittest
import sys
import random 

import rvpy

sys.path.append('..')

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
        eps = 1e-5
        V_height = 1 / (b - a)

        self.assertEqual(self.V.pdf(a - eps), 0)
        self.assertEqual(self.V.pdf((a + b) / 2), V_height)
        self.assertEqual(self.V.pdf(b + eps), 0)

    def test_cunif_cdf(self):
        eps = 1e-5

        # Standard uniform is identity line from 0 to 1
        self.assertEqual(self.U.cdf(0 - eps), 0)
        self.assertEqual(self.U.cdf(1 + eps), 1)

        for i in range(5):
            r = random.random()
            self.assertEqual(self.U.cdf(r), r)

        # Nonstandard uniform is line from a to b
        a = self.V.a
        b = self.V.b
        V_height = 1 / (b - a)

        self.assertEqual(self.V.cdf(a - eps), 0)
        self.assertEqual(self.V.cdf(b + eps), 1)
        for i in range(5):
            r = a + random.random()*(b - a)
            self.assertEqual(self.V.cdf(r), (r - a)/(b - a))

    def test_cunif_conversion(self):
        B = self.U**3

        self.assertIsInstance(self.U.to_beta(), rvpy.Beta)
        self.assertIsInstance(B, rvpy.Beta)

        self.assertAlmostEqual(B.alpha, 1/3)
        self.assertAlmostEqual(B.beta, 1)

    def test_cunif_add_sub(self):
        c = random.randint(1, 11)
        d = random.randint(-11, -1)

        # Adding shifts to the right
        Vplus = self.V + c
        self.assertAlmostEqual(Vplus.a, self.V.a + c)
        self.assertAlmostEqual(Vplus.b, self.V.b + c)
        self.assertAlmostEqual(Vplus.mean, self.V.mean + c)
        self.assertAlmostEqual(Vplus.median, self.V.median + c)
        self.assertAlmostEqual(Vplus.var, self.V.var)
        self.assertAlmostEqual(Vplus.std, self.V.std)

        # Subtracting has same results
        Vminus = self.V + d
        self.assertAlmostEqual(Vminus.a, self.V.a + d)
        self.assertAlmostEqual(Vminus.b, self.V.b + d)
        self.assertAlmostEqual(Vminus.mean, self.V.mean + d)
        self.assertAlmostEqual(Vminus.median, self.V.median + d)
        self.assertAlmostEqual(Vminus.var, self.V.var)
        self.assertAlmostEqual(Vminus.std, self.V.std)

    def test_cunif_mul_div(self):
        c = random.randint(1, 11)
        d = random.randint(1, 11)

        # Multiplication for nonstandard cuniform
        Vmult = c*self.V
        self.assertAlmostEqual(Vmult.a, c*self.V.a)
        self.assertAlmostEqual(Vmult.b, c*self.V.b)
        self.assertAlmostEqual(Vmult.mean, c*self.V.mean)
        self.assertAlmostEqual(Vmult.var, (c**2)*self.V.var)

        # Division does the same inward
        Vdiv = self.V / d
        self.assertAlmostEqual(Vdiv.a, self.V.a / d)
        self.assertAlmostEqual(Vdiv.b, self.V.b / d)
        self.assertAlmostEqual(Vdiv.mean, self.V.mean / d)
        self.assertAlmostEqual(Vdiv.var, self.V.var / (d**2))

    def test_cunif_errors(self):
        # Assert a < b
        with self.assertRaises(AssertionError): rvpy.CUniform(2, 1)
        with self.assertRaises(AssertionError): rvpy.CUniform(1, 1)
        with self.assertRaises(TypeError): self.V.to_beta()
        with self.assertRaises(TypeError): self.V**3

        # Can't add other dists to cuniform
        other_dists = [
            rvpy.Bernoulli(0.3),
            rvpy.Binomial(5, 0.4),
            rvpy.CUniform(),
            rvpy.F(3, 5),
            rvpy.Gamma(3, 2),
            rvpy.Exponential(5),
            rvpy.ChiSq(3),
            rvpy.T(3)
        ]
        for od in other_dists:
            with self.assertRaises(TypeError):
                W = od
                self.V + W

