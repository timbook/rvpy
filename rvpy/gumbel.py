import numpy as np
from scipy.stats import gumbel_r
from . import distribution
from . import logistic

class Gumbel(distribution.Distribution):
    """
    Gumbel Distribution using the following parameterization:

    f(x | mu, beta) = 1/beta * exp(-(z + exp(-z)))
    
    where z = (x - mu) / beta

    Parameters
    ----------
    mu : float, positive
        Location parameter
    beta : float, positive
        Scale parameter

    Methods
    -------
    None

    Relationships
    -------------
    Let X, Y be Gumbel with the same beta. Then:
    * X - Y is Logistic 
    """
    def __init__(self, mu, beta):
        """
        Parameters
        ----------
        mu : float, positive
            Location parameter
        beta : float, positive
            Scale parameter
        """
        assert beta > 0, "scale parameter must be positive"

        # Parameters
        self.mu = mu
        self.beta = beta
        
        # Scipy backend
        self.sp = gumbel_r(loc=mu, scale=beta)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Gumbel(mu={self.mu}, beta={self.beta})"

    def __sub__(self, other):
        if isinstance(other, Gumbel) and self.beta == other.beta:
            return logistic.Logistic(self.mu - other.mu, self.beta)
        elif isinstance(other, Gumbel):
            raise ValueError("To subtract two Gumbels, betas must match")
        else:
            raise TypeError(f"Subtracting something of type {type(other)} from Gumbel not supported")
