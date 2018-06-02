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
            return Normal(new_mu, new_sigma)
        elif isinstance(Y, (int, float)):
            return Normal(self.mu + Y, self.sigma)
        else:
            raise TypeError("Only addition by another (independent) Normal, int, or float supported.")

    def __radd__(self, c):
        if isinstance(c, (int, float)):
            return self + c

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            self.mu *= c
            self.sigma *= c
        else:
            raise TypeError("Only multiplicated by int or float supported.")

    def __rmul__(self, c):
        return self.__mul__(c)

    def __truediv__(self, c):
        if c != 0:
            return self.__mul__(1 / c)
        else:
            raise ZeroDivisionError("Cannot divide a Normal by zero!")

    def __neg__(self):
        return Normal(-self.mu, self.sigma)

    def __pos__(self):
        return self

    def sample(self, *shape):
        return np.random.randn(*shape)

    def mgf(self, t):
        return np.exp(t*self.mu + 0.5*(t**2)*(self.var))

    def pdf(self, x):
        coef = 1 / (2 * np.pi * self.var)**0.5
        pow = -0.5 * (x - self.mu)**2 / self.var
        return coef * np.exp(pow)

    def cdf(self, x):
        return self.sp.cdf(x)

    def prob_interval(self, a, b):
        return self.cdf(b) - self.cdf(a)

    def to_standard(self):
        if self.mu == 0 and self.sigma == 1:
            return StandardNormal()
        else:
            raise ValueError("Must be Normal(0, 1) to standardize!")

    # TODO:
    # - __iadd__, __imul__, __truediv__, ...
    # __or__ for Bayes priors?
    # __pow__(self, 2) --> ChiSq()
    # __truediv__ --> Cauchy()
    # Remove scipy dependency

class StandardNormal(Normal):
    def __init__(self):
        super().__init__(0, 1)

    def __repr__(self):
        return f"StandardNormal(mu=0, sigma=1)"

    def to_nonstandard(self, mu, sigma):
        return Normal(mu=mu, sigma=sigma)

    # TODO: __pow__(self, 2) --> ChiSq()


