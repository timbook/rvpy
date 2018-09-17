import unittest
import sys
import random

sys.path.append('..')

import rvpy

class FDistTests(unittest.TestCase):
    def setUp(self):
        self.df1 = random.randint(5, 15)
        self.df2 = random.randint(5, 15)
        self.X = rvpy.F(self.df1, self.df2)

    def test_f_moments(self):
        d1 = self.df1
        d2 = self.df2

        self.assertAlmostEqual(self.X.mean, d2 / (d2 - 2))

        varnum = 2 * d2**2 * (d1 + d2 - 2)
        varden = d1 * (d2 - 2)**2 * (d2 - 4)
        self.assertAlmostEqual(self.X.var, varnum / varden)

    def test_f_errors(self):
        with self.assertRaises(AssertionError): rvpy.F(0, 1)
        with self.assertRaises(AssertionError): rvpy.F(1, 0)
        with self.assertRaises(AssertionError): rvpy.F(-1, 1)
        with self.assertRaises(AssertionError): rvpy.F(1, -1)


