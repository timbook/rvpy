import numpy as np
from scipy.stats import nbinom
from . import distribution

class NegativeBinomial(distribution.Distribution):
    """
    Negative Binomial Distribution using the following parameterization:

    f(x | r, p) = (x+r-1 r-1) p**n * (1 - p)**x

    Parameters
    ----------
    r : integer, positive
        Number of successes
    p : float, 0 < p < 1
        Probability of success

    Methods
    -------
    to_geometric()
        Converts self to Geometric if r == 1

    Relationships
    -------------
    Let X, Y be NegativeBinomial. Then:
    * X + Y is NegativeBinomial
    """
    def __init__(self, r, p):
        """
        Parameters
        ----------
        r : integer, positive
            Number of successes
        p : float, 0 < p < 1
            Probability of success
        """
        assert isinstance(r, int) and r > 0, 'r must be positive integer'
        assert p > 0 and p < 1, 'p must be a number between 0 and 1'

        # Parameters
        self.r = r
        self.p = p
        self.q = 1 - p

        # Scipy backend
        self.sp = nbinom(r, p)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"NegativeBinomial(r={self.r}, p={self.p})"

    def __add__(self, other):
        if isinstance(other, NegativeBinomial) and other.p == self.p:
            return NegativeBinomial(self.r + other.r, self.p)
        else:
            raise TypeError("Can only add Geometric or NegativeBinomial to NegativeBinomial")

    def to_geometric(self):
        assert self.r == 1, "r must be 1 to cast to negative binomial"
        return Geometric(p=self.p)

class Geometric(NegativeBinomial):
    """
    Geometric Distribution using the following parameterization:

    f(x | p) = p * (1 - p)**x

    Parameters
    ----------
    p : float, 0 < p < 1
        Probability of success

    Methods
    -------
    to_negative_binomial()
        Converts self to NegativeBinomial

    Relationships
    -------------
    Let X, Y be Geometric. Then:
    * X + Y is NegativeBinomial
    """
    def __init__(self, p):
        """
        Parameters
        ----------
        p : float, 0 < p < 1
            Probability of success
        """
        super().__init__(r=1, p=p)

    def __repr__(self):
        return f"Geometric(p={self.p})"

    def to_negative_binomial(self):
        return NegativeBinomial(r=1, p=self.p)
