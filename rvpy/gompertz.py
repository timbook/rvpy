import numpy as np
from scipy.stats import gompertz
from . import distribution

class Gompertz(distribution.Distribution):
    """
    Gompertz Distribution using the following parameterization:

    f(x | eta, b) = b * eta * exp(bx) * exp(-eta * exp(bx))

    Parameters
    ----------
    eta : float, positive
        Shape parameter
    b : float, positive
        Inverse scale parameter

    Methods
    -------
    None

    Relationships
    -------------
    None implemented
    """
    def __init__(self, eta, b):
        """
        Parameters
        ----------
        eta : float, positive
            Shape parameter
        b : float, positive
            Inverse scale parameter
        """
        assert eta > 0, "eta must be positive"
        assert b > 0, "b must be positive"

        # Parameters
        self.eta = eta
        self.b = b

        # Scipy backend
        self.sp = gompertz(c=eta, scale=1/b)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Gompertz(eta={self.eta}, b={self.b})"
