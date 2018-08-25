import numpy as np
from scipy.stats import norm
from . import distribution

class Normal(distribution.Distribution):
    def __init__(self, mu=0, sigma=1):
        assert isinstance(mu, (int, float)), "mu must be numeric!"
        assert isinstance(sigma, (int, float)), "sigma must be numeric!"
        assert sigma > 0, "sigma must be positive"

        self.mu = mu
        self.sigma = sigma

        # Scipy backend
        self.sp = norm(mu, sigma)

        # Intialize super - does nothing yet.
        super().__init__()

    def __repr__(self):
        return f"Normal(mu={self.mu}, sigma={self.sigma})"

    def __add__(self, other):
        if isinstance(other, Normal):
            new_mu = other.mu + self.mu
            new_sigma = (self.var + other.var)**0.5
            return Normal(new_mu, new_sigma)
        elif isinstance(other, (int, float)):
            return Normal(self.mu + other, self.sigma)
        else:
            raise TypeError("Only addition by another (independent) Normal, int, or float supported.")

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Normal(other*self.mu, float(np.abs(other)*self.sigma))
        else:
            raise TypeError("Only multiplicated by int or float supported.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)) and other != 0:
            return self.__mul__(1 / other)
        else:
            raise ZeroDivisionError("Cannot divide a Normal by zero!")

    def __neg__(self):
        return Normal(-self.mu, self.sigma)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def mgf(self, t):
        return np.exp(t*self.mu + 0.5*(t**2)*(self.var))

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

class StandardNormal(Normal):
    def __init__(self):
        # Get non-standard Normal distribution initialization
        super().__init__(0, 1)

    def __repr__(self):
        return f"StandardNormal(mu=0, sigma=1)"

    def to_nonstandard(self):
        return Normal(mu=0, sigma=1)

    # TODO: __pow__(self, 2) --> ChiSq()


