import unittest
import sys
import random

sys.path.append('..')

import rvpy

class TDistTests(unittest.TestCase):
    def setUp(self):
        self.nu = random.randint(10, 20)
        self.X = rvpy.T(self.nu)

        self.T1 = rvpy.T(1)
        self.T2 = rvpy.T(2)
        self.T3 = rvpy.T(3)

    def test_t_moments(self):
        self.assertAlmostEqual(self.X.mean, 0)
        self.assertAlmostEqual(self.X.median, 0)
        self.assertAlmostEqual(self.X.var, self.nu / (self.nu - 2))
        self.assertAlmostEqual(self.X.skew, 0)
        self.assertAlmostEqual(self.X.kurtosis, 6 / (self.nu - 4))
        
        # TODO: Moments for lower dfs

    def test_t_pdf(self):
        self.assertAlmostEqual(self.X.pdf(-15), 0)
        self.assertAlmostEqual(self.X.pdf(15), 0)

    def test_t_cdf(self):
        self.assertAlmostEqual(self.X.cdf(-15), 0)
        self.assertAlmostEqual(self.X.cdf(0), 1/2)
        self.assertAlmostEqual(self.X.cdf(15), 1)

    def test_t_conversion(self):
        self.assertIsInstance(self.X, rvpy.T)
        
        Xconv = self.X**2
        self.assertIsInstance(Xconv, rvpy.F)
        self.assertEqual(Xconv.df1, 1)
        self.assertEqual(Xconv.df2, self.nu)

        Xconvinv = self.X**-2
        self.assertIsInstance(Xconvinv, rvpy.F)
        self.assertEqual(Xconvinv.df1, self.nu)
        self.assertEqual(Xconvinv.df2, 1)

        # TODO: .to_cauchy()

    def test_t_errors(self):
        with self.assertRaises(AssertionError): self.X**3
        with self.assertRaises(AssertionError): rvpy.T(0)
        with self.assertRaises(AssertionError): rvpy.T(-1)
