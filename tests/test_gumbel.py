import unittest
import sys
import random 

import numpy as np

import rvpy

sys.path.append('..')

class GumbelTests(unittest.TestCase):
    def setUp(self):
        self.mu = random.randint(-10, 10)
        self.beta = 10*random.random()
        self.X = rvpy.Gumbel(self.mu, self.beta)

    def test_gumbel_moments(self):
        self.assertAlmostEqual(self.X.mean, self.mu + self.beta*np.euler_gamma)
        self.assertAlmostEqual(self.X.var, np.pi**2 * self.beta**2 / 6)
        self.assertAlmostEqual(self.X.median, self.mu - self.beta*np.log(np.log(2)))
        self.assertAlmostEqual(self.X.kurtosis, 2.4)

    def test_gumbel_pdf(self):
        x = random.random()
        z = (x - self.mu) / self.beta
        self.assertAlmostEqual(
            self.X.pdf(x),
            1/self.beta * np.exp(-(z + np.exp(-z)))
        )

    def test_gumbel_add_sub(self):
        m1 = random.random()
        b = random.random()
        m2 = random.random()
        A = rvpy.Gumbel(m1, b)
        B = rvpy.Gumbel(m2, b)
        C = A - B

        self.assertEqual(C.loc, m1 - m2)
        self.assertEqual(C.scale, b)

    def test_gumbel_errors(self):
        with self.assertRaises(AssertionError): rvpy.Gumbel(0, 0)
        with self.assertRaises(AssertionError): rvpy.Gumbel(0, -1)
