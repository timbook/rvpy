import numpy as np
from scipy.stats import laplace
from . import distribution, gamma

class Laplace(distribution.Distribution):
    """
    Laplace Distribution using the following parameterization:

    f(x | mu, b) = 1 / 2b * exp(-|x - mu| / b)

    Parameters
    ----------
    mu : float
        Location and mean parameter
    b : float, positive
        Scale parameter

    Methods
    -------
    abs()
        Returns |self|, which is Exponential

    Relationships
    -------------
    Let X be Laplace, c float. Then:
    * X + c is Laplace
    * cX is Laplace
    """
    def __init__(self, mu=0, b=1):
        """
        Parameters
        ----------
        mu : float
            Location and mean parameter
        b : float, positive
            Scale parameter
        """
        assert b > 0, "b must be positive"

        # Parameters
        self.mu = mu
        self.b = b

        # Scipy backend
        self.sp = laplace(loc=mu, scale=b)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Laplace(mu={self.mu}, b={self.b})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Laplace(self.mu + other, self.b)
        else:
            raise TypeError(f"Can't add objects of type {type(other)} to Laplace")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Laplace(other*self.mu, abs(other)*self.b)
        else:
            raise TypeError(f"Can't multiply or divide objects of type {type(other)} to Laplace")

    def __truediv__(self, other):
        return self.__mul__(1/other)

    def abs(self):
        assert self.mu == 0, "Must have mu == 0 for conversion to Exponential"
        return gamma.Exponential(self.b)
