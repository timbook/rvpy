import numpy as np
from scipy.stats import poisson
from . import distribution

class Poisson(distribution.Distribution):
    def __init__(self, mu):
        assert isinstance(mu, int), "mu must be an integer"
        assert mu > 0, "mu must be positive integer"

        # Parameters
        self.mu = mu

        # Scipy backend
        self.sp = poisson(mu)

        # Intialize super
        super().__init__()

    def __repr__(self):
        return f"Poisson(mu={self.mu})"

    def __add__(self, Y):
        if isinstance(Y, Poisson):
            return Poisson(self.mu + Y.mu)
        else:
            raise TypeError


