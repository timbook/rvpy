import unittest
import sys
import random 

import rvpy

sys.path.append('..')

class ParetoTests(unittest.TestCase):
    def setUp(self):
        alpha = random.randint(1, 11)
        beta = 2 + random.expovariate(1/10)
        self.X = rvpy.Pareto(alpha, beta)

    def test_pareto_moments(self):
        a = self.X.alpha
        b = self.X.beta

        self.assertAlmostEqual(self.X.mean, (b * a) / (b - 1))
        self.assertAlmostEqual(self.X.var, (b * a**2) / ((b - 1)**2 * (b - 2)))

    def test_pareto_pdf(self):
        self.assertEqual(self.X.pdf(self.X.alpha - 0.01), 0)
        self.assertGreater(self.X.pdf(self.X.alpha + 0.01), 0)

    def test_pareto_cdf(self):
        self.assertEqual(self.X.cdf(self.X.alpha - 1), 0)
        self.assertGreater(self.X.cdf(self.X.alpha + 1), 0)
    
    def test_pareto_conversion(self):
        # TODO: relationship with exponential distn
        pass

    def test_pareto_errors(self):
        with self.assertRaises(AssertionError): rvpy.Pareto(0, 2)
        with self.assertRaises(AssertionError): rvpy.Pareto(2, 0)
        with self.assertRaises(AssertionError): rvpy.Pareto(-2, 2)
        with self.assertRaises(AssertionError): rvpy.Pareto(2, -2)
