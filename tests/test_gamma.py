import unittest
import sys
import random

sys.path.append('..')

import rvpy

class GammaTests(unittest.TestCase):
    def setUp(self):
        self.alpha1 = random.expovariate(1/10)
        self.alpha2 = random.expovariate(1/10)
        self.beta = random.expovariate(1/10)

        self.X = rvpy.Gamma(self.alpha1, self.beta)
        self.Y = rvpy.Gamma(self.alpha2, self.beta)

    def test_gamma_moments(self):
        self.assertAlmostEqual(self.X.mean, self.alpha1*self.beta)
        self.assertAlmostEqual(self.X.var, self.alpha1*self.beta**2)
        self.assertAlmostEqual(self.X.skew, 2/self.alpha1**0.5)
        self.assertAlmostEqual(self.X.kurtosis, 6/self.alpha1)

    def test_gamma_conversion(self):
        lambd = random.expovariate(1/5)
        U = rvpy.Gamma(1, lambd)
        self.assertIsInstance(U.to_exponential(), rvpy.Exponential)
        self.assertEqual(U.to_exponential().mean, lambd)

        alph = 2*random.randrange(3, 11)
        V = rvpy.Gamma(alph, 2)
        self.assertIsInstance(V.to_chisq(), rvpy.ChiSq)
        self.assertEqual(V.to_chisq().df, 2*alph)

    def test_gamma_add_sub(self):
        Z = self.X + self.Y
        self.assertEqual(Z.mean, (self.alpha1 + self.alpha2)*self.beta)
        self.assertEqual(Z.alpha, self.alpha1 + self.alpha2)
        self.assertEqual(Z.beta, self.beta)

    def test_gamma_mul_div(self):
        c = random.expovariate(1/10)

        Z1 = c*self.X
        Z2 = self.X*c
        Z3 = self.X / c

        self.assertAlmostEqual(Z1.beta, c*self.X.beta)
        self.assertAlmostEqual(Z2.beta, c*self.X.beta)
        self.assertAlmostEqual(Z3.beta, self.X.beta / c)

    def test_gamma_errors(self):
        with self.assertRaises(AssertionError): rvpy.Gamma(0, 1)
        with self.assertRaises(AssertionError): rvpy.Gamma(1, 0)
        with self.assertRaises(AssertionError): rvpy.Gamma(-1, 1)
        with self.assertRaises(AssertionError): rvpy.Gamma(1, -1)
        with self.assertRaises(TypeError): rvpy.Gamma(3, 4) + 3

class ExponentialTests(unittest.TestCase):
    def setUp(self):
        self.beta1 = random.expovariate(1/10)
        self.beta2 = random.expovariate(1/10)
        self.alpha1 = random.randint(2, 10)
        
        self.E1 = rvpy.Exponential(self.beta1)
        self.E2 = rvpy.Exponential(self.beta1)
        self.G1 = rvpy.Gamma(self.alpha1, self.beta1)

    def test_exp_moments(self):
        self.assertEqual(self.E1.mean, self.beta1)
        self.assertEqual(self.E1.mean, self.E1.scale)
        self.assertEqual(self.E1.std, self.E1.scale)
        self.assertEqual(self.E1.var, self.beta1**2)
        self.assertEqual(self.E1.skew, 2)
        self.assertEqual(self.E1.kurtosis, 6)

    def test_exp_conversion(self):
        E4 = rvpy.Exponential(2)

        self.assertIsInstance(self.E1.to_gamma(), rvpy.Gamma)
        self.assertIsInstance(E4.to_chisq(), rvpy.ChiSq)

        self.assertEqual(self.E1.to_gamma().mean, self.E1.mean)
        self.assertEqual(self.E1.to_gamma().var, self.E1.var)
        self.assertEqual(self.E1.to_gamma().alpha, 1)
        self.assertEqual(self.E1.to_gamma().beta, self.beta1)
        self.assertEqual(E4.to_chisq().df, 2)

    def test_exp_add_sub(self):
        G = self.E1 + self.E2
        self.assertIsInstance(G, rvpy.Gamma)
        self.assertEqual(G.alpha, 2)
        self.assertEqual(G.beta, self.E1.scale)

        H = self.E1 + self.G1
        self.assertIsInstance(H, rvpy.Gamma)
        self.assertEqual(H.alpha, self.G1.alpha + 1)
        self.assertEqual(H.beta, self.G1.beta)
        self.assertEqual(H.beta, self.E1.scale)

        # TODO: E1 - E2 --> Laplace

    def test_exp_mul_div(self):
        c = random.expovariate(1/10)
        Emul = c * self.E1
        Ediv = self.E1 / c

        self.assertIsInstance(Emul, rvpy.Exponential)
        self.assertAlmostEqual(Emul.mean, c*self.E1.mean)

        self.assertIsInstance(Ediv, rvpy.Exponential)
        self.assertAlmostEqual(Ediv.mean, self.E1.mean / c)

    def test_exp_errors(self):
        with self.assertRaises(AssertionError): rvpy.Exponential(0)
        with self.assertRaises(AssertionError): rvpy.Exponential(-1)
        with self.assertRaises(TypeError): rvpy.Exponential(3) + 2
        with self.assertRaises(ValueError):
            rvpy.Exponential(3) + rvpy.Exponential(4)
        with self.assertRaises(ValueError):
            rvpy.Exponential(3) + rvpy.Gamma(2, 4)
        with self.assertRaises(AssertionError):
            rvpy.Exponential(3).to_chisq()

class ChiSqTests(unittest.TestCase):
    def setUp(self):
        self.C1 = rvpy.ChiSq(2)
        self.C2 = rvpy.ChiSq(random.randint(3, 10))

    def test_chisq_moments(self):
        self.assertEqual(self.C1.mean, self.C1.df)
        self.assertEqual(self.C1.var, 2*self.C1.df)
        self.assertAlmostEqual(self.C1.skew, (8/self.C1.df)**0.5)
        self.assertAlmostEqual(self.C1.kurtosis, 12/self.C1.df)

    def test_chisq_conversion(self):
        Cexp = self.C1.to_exponential()
        self.assertIsInstance(Cexp, rvpy.Exponential)
        self.assertEqual(Cexp.scale, 2)

        Cgamma = self.C2.to_gamma()
        self.assertIsInstance(Cgamma, rvpy.Gamma)
        self.assertEqual(Cgamma.alpha, self.C2.df/2)
        self.assertEqual(Cgamma.beta, 2)

    def test_chisq_add_sub(self):
        Cadd = self.C1 + self.C2
        self.assertIsInstance(Cadd, rvpy.ChiSq)
        self.assertEqual(Cadd.df, self.C1.df + self.C2.df)

        Z = rvpy.StandardNormal()
        Cpnorm = self.C2 + Z**2
        self.assertIsInstance(Cpnorm, rvpy.ChiSq)
        self.assertEqual(Cpnorm.df, self.C2.df + 1)

    def test_chisq_mul_div(self):
        c = random.expovariate(1/10)

        Cmul = c * self.C2
        self.assertIsInstance(Cmul, rvpy.Gamma)
        self.assertNotIsInstance(Cmul, rvpy.ChiSq)
        self.assertEqual(Cmul.alpha, self.C2.df / 2)
        self.assertEqual(Cmul.beta, 2*c)

        Cdiv = self.C2 / c
        self.assertIsInstance(Cdiv, rvpy.Gamma)
        self.assertNotIsInstance(Cdiv, rvpy.ChiSq)
        self.assertEqual(Cmul.alpha, self.C2.df / 2)
        self.assertEqual(Cdiv.beta, 2/c)

    def test_chisq_errors(self):
        with self.assertRaises(AssertionError): rvpy.ChiSq(0)
        with self.assertRaises(AssertionError): rvpy.ChiSq(-1)
        with self.assertRaises(AssertionError): rvpy.ChiSq(0.5)
        with self.assertRaises(TypeError): rvpy.ChiSq(3) + 2
        with self.assertRaises(ValueError):
            rvpy.ChiSq(3) + rvpy.Exponential(4)
        with self.assertRaises(AssertionError):
            rvpy.ChiSq(3).to_exponential()







