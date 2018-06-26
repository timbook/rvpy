import numpy as np
from scipy.stats import binom
from . import distribution, bernoulli

class Binomial(distribution.Distribution):
    def __init__(self, n, p):
        assert p < 1 and p > 0, "p must be a float between 0 and 1"
        assert isinstance(p, float), "p must be a float"
        assert isinstance(n, int), "n must be an integer"

        super().__init__()

        # Parameters
        q = 1 - p
        self.p = p
        self.q = q
        self.n = n

        # Moments
        self.mean = n*p
        self.std = (n*p*q)**0.5
        self.var = n*p*q
        self.skew = (1 - 2*p) / np.sqrt(n*p*q)
        self.kurtosis = (1 - 6*p*q) / (n*p*q) 

        # Scipy backend
        self.sp = binom(n, p)

    def __repr__(self):
        return f"Binomial(n={self.n}, p={self.p})"

    def __add__(self, Y):
        assert isinstance(Y, Binomial), "can only add Binomials to Binomials"
        assert self.p == Y.p, "values of p must match in order to add"
        return Binomial(self.n + Y.n, self.p)

    def sample(self, *shape):
        return np.random.binomial(n=self.n, p=self.p, size=shape)

    def pmf(self, k):
        assert k >= 0 and k <= self.n, "k must be between 0 and n"
        assert isinstance(k, int), "k must be an integer"
        return self.sp.pmf(k)

    def cdf(self, k):
        assert isinstance(k, int), "k must be an integer"
        return self.sp.cdf(k)

    def mgf(self, t):
        return (self.q + self.p * np.exp(t))**self.n

    def to_bernoulli(self):
        assert self.n == 1, "Must have n == 1 to convert to downcast to Bernoulli"
        return bernoulli.Bernoulli(self.p)
