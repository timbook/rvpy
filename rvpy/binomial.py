import numpy as np
from scipy.stats import binom
from . import distribution, bernoulli

class Binomial(distribution.Distribution):
    def __init__(self, n, p):
        assert p < 1 and p > 0, "p must be a float between 0 and 1"
        assert isinstance(p, float), "p must be a float"
        assert isinstance(n, int), "n must be an integer"

        # Parameters
        q = 1 - p
        self.p = p
        self.q = q
        self.n = n

        # Scipy backend
        self.sp = binom(n, p)

        # Intialize super - does nothing yet.
        super().__init__()

        # Get moments
        super()._moments()

    def __repr__(self):
        return f"Binomial(n={self.n}, p={self.p})"

    def __add__(self, Y):
        assert isinstance(Y, Binomial), "can only add Binomials to Binomials"
        assert self.p == Y.p, "values of p must match in order to add"
        return Binomial(self.n + Y.n, self.p)

    def mgf(self, t):
        return (self.q + self.p * np.exp(t))**self.n

    def to_bernoulli(self):
        assert self.n == 1, "Must have n == 1 to convert to downcast to Bernoulli"
        return bernoulli.Bernoulli(self.p)
