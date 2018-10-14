import numpy as np
from scipy.stats import binom
from . import distribution

class Binomial(distribution.Distribution):
    def __init__(self, n, p):
        assert p < 1 and p > 0, "p must be a float between 0 and 1"
        assert n > 0, "n must be a positive integer"
        assert isinstance(p, float), "p must be a float"
        assert isinstance(n, int), "n must be an integer"

        # Parameters
        self.p = p
        self.q = 1 - p
        self.n = n

        # Scipy backend
        self.sp = binom(n, p)

        # Intialize super
        super().__init__()

    def __repr__(self):
        return f"Binomial(n={self.n}, p={self.p})"

    def __add__(self, Y):
        if isinstance(Y, Binomial) and self.p == Y.p:
            return Binomial(self.n + Y.n, self.p)
        else:
            raise TypeError("Can only add Binomials or Benoulli to Binomials")

    def mgf(self, t):
        return (self.q + self.p * np.exp(t))**self.n

    def to_bernoulli(self):
        assert self.n == 1, \
                "Must have n == 1 to convert to downcast to Bernoulli"
        return Bernoulli(self.p)

class Bernoulli(Binomial):
    def __init__(self, p):
        # Get Bernoulli distribution initialization
        super().__init__(n=1, p=p)

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    def to_binomial(self):
        return Binomial(n=1, p=self.p)
