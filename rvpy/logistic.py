import numpy as np
from scipy.stats import logistic
from . import distribution

class Logistic(distribution.Distribution):
    """
    Logistic Distribution using the following parameterization:

    f(x | loc, scale) = 

    Parameters
    ----------
    loc : float, positive
        Location parameter
    scale : float, positive
        Scale parameter

    Methods
    -------
    None

    Relationships
    -------------
    Let X be Logistic, a, b float. Then:
    * aX + b is Logistic
    * exp(X) is Log-Logistic (not yet implemented)
    """
    def __init__(self, loc=0, scale=1):
        """
        Parameters
        ----------
        loc : float, positive
            Location parameter
        scale : float, positive
            Scale parameter
        """
        assert scale > 0, "scale parameter must be positive"

        # Parameters
        self.loc = loc
        self.scale = scale

        # Scipy backend
        self.sp = logistic(loc=loc, scale=scale)

        super().__init__()

    def __repr__(self):
        return f"Logistic(loc={self.loc}, scale={self.scale})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Logistic(self.loc + other, self.scale)
        else:
            raise TypeError(f"Can't add or subtract objects of type {type(other)} to Logistic")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Logistic(other * self.loc, other * self.scale)
        else:
            raise TypeError(f"Can't multiply objects of type {type(other)} by Logistic")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1/other)
        else:
            raise TypeError(f"Can't divide objects of type {type(other)} by Logistic")

    # TODO: Gumbel - Gumbel = Logistic
    # TODO: exp(Logistic) = Log-Logistic


