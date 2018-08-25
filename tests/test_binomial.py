import unittest
import sys
import random

sys.path.append('..')

import rvpy

class BinomialTests(unittest.TestCase):
    def setUp(self):
        self.p1 = random.random()
        self.p2 = random.random()
        self.p3 = random.random()
        self.n1 = random.randrange(1, 10)
        self.n2 = random.randrange(1, 10)
        self.X1 = rvpy.Binomial(self.n1, self.p1)
        self.X2 = rvpy.Binomial(self.n2, self.p2)
        self.Y = rvpy.Bernoulli(self.p3)

    def test_bin_moments(self):
        pass

    def test_bin_pmf(self):
        self.assertAlmostEqual(self.Y.pmf(0) + self.Y.pmf(1), 1)
        # TODO: This

    def test_bin_cdf(self):
        self.assertEqual(self.X1.cdf(self.X1.n), 1)
        self.assertEqual(self.X1.cdf(0), (1 - self.X1.p)**self.X1.n)
        # TODO: More of these
