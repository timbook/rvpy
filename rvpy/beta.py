import numpy as np
from scipy.stats import beta as bta
from . import distribution
from . import cuniform

class Beta(distribution.Distribution):
    """
    Beta Distribution using the following parameterization:

    f(x | alpha, beta) = 1 / B(alpha, beta) * x**(alpha - 1) * x**(beta - 1)

    Parameters
    ----------
    alpha : float, positive
        Shape parameter
    beta : float, positive
        Shape parameter

    Methods
    -------
    to_cuniform()
        Converts self to CUniform if (alpha, beta) == (1, 1)

    Relationships
    -------------
    Let X be Beta(1, 1). Then:
    * X is CUniform(0, 1)
    """
    def __init__(self, alpha, beta):
        """
        Parameters
        ----------
        alpha : float, positive
            Shape parameter
        beta : float, positive
            Shape parameter
        """
        assert alpha > 0 and beta > 0, "alpha and beta must be positive"

        # Parameters
        self.alpha = alpha
        self.beta = beta

        # Scipy backend
        self.sp = bta(alpha, beta)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Beta(alpha={self.alpha}, beta={self.beta})"

    # TODO: Implement crazy MGF for Beta.

    def to_cuniform(self):
        assert self.alpha == 1 and self.beta == 1, "Alpha and beta must be equal to 1 to cast to CUniform"
        return cuniform.CUniform(a=0, b=1)


