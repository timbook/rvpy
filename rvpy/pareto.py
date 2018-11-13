import numpy as np
from scipy.stats import pareto
from . import distribution

class Pareto(distribution.Distribution):
    """
    Pareto Distribution using the following parameterization:

    f(x | alpha, beta) = beta * alpha**beta / x**(beta + 1)

    Parameters
    ----------
    alpha : float, positive
        Scale parameter
    beta : float, positive
        Shape parameter

    Methods
    -------
    None

    Relationships
    -------------
    Let X be Pareto with alpha = 1. Then:
    * log(X) is exponential
    """
    def __init__(self, alpha, beta):
        """
        Parameters
        ----------
        alpha : float, positive
            Scale parameter
        beta : float, positive
            Shape parameter
        """
        assert alpha > 0 and beta > 0, \
                "alpha and beta parameters must be positive"

        # Parameters
        self.alpha = alpha
        self.beta = beta

        # Scipy backend
        self.sp = pareto(b=beta, scale=alpha)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Pareto(alpha={self.alpha}, beta={self.beta})"

    # TODO: log/exp relationship with Exponential
