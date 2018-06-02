import numpy as np
from scipy.stats import norm

class Normal:
    def __init__(self, mu=0, sigma=1):
        # Parameters
        self.mu = mu
        self.sigma = sigma
        
        # Moments
        self.mean = mu
        self.std = sigma
        self.var = sigma**2
        self.skew = 0
        self.kurtosis = 0

        # Scipy backend
        self.sp = norm(mu, sigma)

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    def __add__(self, Y):
        if isinstance(Y, Normal):
            new_mu = self.mu + Y.mu
            new_sigma = (self.var + Y.var)**0.5
            return Normal(new_mu, new_sigma)
        elif isinstance(Y, (int, float)):
            return Normal(self.mu + Y, self.sigma)
        else:
            raise TypeError

    def __radd__(self, c):
        if isinstance(c, (int, float)):
            return self + c

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return Normal(c * self.mu, c * self.sigma)
        else:
            raise TypeError

    def __rmul__(self, c):
        return self.__mul__(c)

    def sample(self, *shape):
        return np.random.randn(*shape)

    def mgf(self, t):
        return np.exp(t*self.mu + 0.5*(t**2)*(self.var))

    def pdf(self, x):
        return self.sp.pdf(x)

    def cdf(self, x):
        return self.sp.cdf(x)

    def prob_interval(self, a, b):
        return self.cdf(b) - self.cdf(a)

    # TODOs:
    # - __iadd__, __imul__, __truediv__, ...
    # __or__ for Bayes priors?
    # __pow__(self, 2) --> ChiSq()
    # __truediv__ --> Cauchy()
    # Remove scipy dependency
