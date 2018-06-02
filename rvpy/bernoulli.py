import numpy as np

class Bernoulli:
    def __init__(self, p):
        assert isinstance(p, float) and p < 1 and p > 0, "p must be a float between 0 and 1"
        # Parameters
        q = 1 - p
        self.p = p
        self.q = q

        # Moments
        self.std = (p*q)**0.5
        self.mean = p
        self.var = p*q
        self.skew = (1 - 2*p) / np.sqrt(p*q)
        self.kurtosis = (1 - 6*p*q) / (p*q) 

        # Other properties
        # self.median = _
        # self.mode = _
        # self.entropy = _

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    def __pos__(self):
        return self

    # TODO: __add__ to Binomials

    def sample(self, *shape):
        return np.random.binomial(n=1, p=self.p, size=shape)

    def pdf(self, x):
        assert x in [0, 1], "Bernoulli has support {0, 1}"
        return x*self.p + (1 - x)*self.q

    def cdf(self, x):
        if x < 0:
            return 0
        elif x < 1:
            return self.q
        else:
            return 1

    def mgf(self, t):
        return self.q + self.p * np.exp(t)

    def to_binomial(self):
        # TODO: this
        # return Binomial(n=1, p=self.p)
        pass
