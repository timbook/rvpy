import unittest
import sys
import random

sys.path.append('..')

import rvpy

class NormalTests(unittest.TestCase):
    def setUp(self):
        self.rnd1 = random.random()
        self.rnd2 = random.random()
        self.Z = rvpy.StandardNormal()
        self.X = rvpy.Normal(mu=self.rnd1, sigma=self.rnd2)

    def test_normal_moments(self):
        # Standard normal
        self.assertEqual(self.Z.mean, 0)
        self.assertEqual(self.Z.std, 1)
        self.assertEqual(self.Z.var, 1)
        self.assertEqual(self.Z.skew, 0)
        self.assertEqual(self.Z.kurtosis, 0)

        # Nonstandard normal
        self.assertEqual(self.X.mean, self.rnd1)
        self.assertEqual(self.X.std, self.rnd2)
        self.assertEqual(self.X.var, self.rnd2**2)
        self.assertEqual(self.X.skew, 0)
        self.assertEqual(self.X.kurtosis, 0)

    def test_normal_pdf(self):
        # pdf should be zero far from mean
        self.assertAlmostEqual(self.Z.pdf(10), 0)
        self.assertAlmostEqual(self.Z.pdf(-10), 0)
        self.assertAlmostEqual(self.X.pdf(10), 0)
        self.assertAlmostEqual(self.X.pdf(-10), 0)

        # pdf should be symmetric
        c = random.random()
        self.assertAlmostEqual(
            self.X.pdf(self.X.mean - c),
            self.X.pdf(self.X.mean + c)
        )

        self.assertAlmostEqual(
            self.Z.pdf(self.Z.mean - c),
            self.Z.pdf(self.Z.mean + c)
        )

    def test_normal_cdf(self):
        # CDF bounds and medians
        self.assertAlmostEqual(self.Z.cdf(10), 1)
        self.assertAlmostEqual(self.Z.cdf(-10), 0)
        self.assertAlmostEqual(self.Z.cdf(0), 0.5)
        self.assertAlmostEqual(self.X.cdf(10), 1)
        self.assertAlmostEqual(self.X.cdf(-10), 0)
        self.assertAlmostEqual(self.X.cdf(self.rnd1), 0.5)

    def test_normal_conversion(self):
        # Empty parameters should give N(0, 1), which converts to Std Norm
        X = rvpy.Normal()
        self.assertIsInstance(X.to_standard(), rvpy.StandardNormal)

        # Clean conversion to/from Std Norm
        self.assertIsInstance(self.Z.to_nonstandard(), rvpy.Normal)
        self.assertIsInstance(self.Z.to_nonstandard().to_standard(), rvpy.Normal)
        self.assertEqual(self.Z.to_nonstandard().mean, 0)
        self.assertEqual(self.Z.to_nonstandard().std, 1)

        # Standardizing produces standard normal
        Z = (self.X - self.X.mean) / self.X.std
        self.assertIsInstance(Z.to_standard(), rvpy.StandardNormal)

    def test_normal_add_sub(self):
        X1 = self.X
        X2 = rvpy.Normal(mu=random.random(), sigma=random.random())

        # Test addition
        X3 = X1 + X2
        new_sigma = (X1.sigma**2 + X2.sigma**2)**0.5
        self.assertAlmostEqual(X3.mean, X1.mu + X2.mu)
        self.assertAlmostEqual(X3.sigma, new_sigma)

        # Test subtraction
        X4 = X1 - X2
        self.assertAlmostEqual(X4.mean, X1.mu - X2.mu)
        self.assertAlmostEqual(X4.sigma, new_sigma)

    def test_normal_mul_div(self):
        c = random.random() + 1

        # Scaling normal
        Y1 = c * self.X
        self.assertAlmostEqual(Y1.mean, c * self.X.mean)
        self.assertAlmostEqual(Y1.var, (c**2) * self.X.var)
        self.assertAlmostEqual(Y1.std, c * self.X.std)

        Y2 = self.X / c
        self.assertAlmostEqual(Y2.mean, self.X.mean / c)
        self.assertAlmostEqual(Y2.var, self.X.var * (c**-2))
        self.assertAlmostEqual(Y2.std, self.X.std * (c**-1))

        # Scaling standard
        Z1 = c * self.Z
        self.assertAlmostEqual(Z1.mean, 0)
        self.assertAlmostEqual(Z1.var, c**2)
        self.assertAlmostEqual(Z1.std, c)

        Z2 = self.Z / c
        self.assertAlmostEqual(Z2.mean, 0)
        self.assertAlmostEqual(Z2.var, c**-2)
        self.assertAlmostEqual(Z2.std, c**-1)

    def test_normal_errors(self):
        # Broken conversion when not N(0, 1)
        with self.assertRaises(ValueError): self.X.to_standard()

        # Negative sigmas not allowed
        with self.assertRaises(AssertionError): rvpy.Normal(0, -1)
