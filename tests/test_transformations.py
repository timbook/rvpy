import unittest
import sys
import rvpy

sys.path.append('..')

class TransformationTests(unittest.TestCase):
    def test_log(self):
        X = rvpy.LogNormal()
        Y = rvpy.log(X)

        self.assertIsInstance(X, rvpy.LogNormal)
        self.assertIsInstance(Y, rvpy.Normal)

    def test_exp(self):
        X = rvpy.Normal()
        Y = rvpy.exp(X)

        self.assertIsInstance(X, rvpy.Normal)
        self.assertIsInstance(Y, rvpy.LogNormal)

    def test_abs(self):
        X = rvpy.Laplace()
        Y = rvpy.abs(X)

        self.assertIsInstance(X, rvpy.Laplace)
        self.assertIsInstance(Y, rvpy.Exponential)

    def test_sqrt(self):
        X = rvpy.LogNormal()
        Y = rvpy.sqrt(X)

        self.assertIsInstance(X, rvpy.LogNormal)
        self.assertIsInstance(Y, rvpy.LogNormal)

    def test_pow(self):
        X = rvpy.StandardNormal()
        Y = rvpy.pow(X, 2)

        self.assertIsInstance(X, rvpy.StandardNormal)
        self.assertIsInstance(Y, rvpy.ChiSq)

        Z = rvpy.LogNormal()
        W = rvpy.pow(Z, .321)

        self.assertIsInstance(Z, rvpy.LogNormal)
        self.assertIsInstance(W, rvpy.LogNormal)
