import numpy as np
from scipy.stats import uniform

from . import distribution

class Degenerate(distribution.Distribution):
    """
    Degenerate Distribution using the following parameterization:

    f(x | k) = 1 if x == k, 0 otherwise

    Parameters
    ----------
    k : float
        Location parameter

    Methods
    -------
    None

    Relationships
    -------------
    Let X be Degenerate, c float. Then:
    * X + c is Degenerate
    """
    def __init__(self, k):
        """
        Parameters
        ----------
        k : float
            Location parameter
        """
        # Parameters
        self.k = k

        # Scipy backend
        self.sp = uniform(loc=k, scale=0)

        # Moments
        self.mean = k
        self.std = 0
        self.var = 0
        self.median = k
        self.skew = np.nan
        self.kurtosis = np.nan
        self.entropy = 0

    def __repr__(self):
        return f"Degenerate(k={self.k})"

    def pdf(self, x):
        return np.where(x == self.k, 1, 0)

    def cdf(self, x):
        return np.where(x < self.k, 0, 1)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Degenerate(self.k + other)
        else:
            raise TypeError("Can only add constants to Degenerate")
