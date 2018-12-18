import numpy as np
from math import log, exp
from scipy.stats import logistic, fisk
from . import distribution

class Logistic(distribution.Distribution):
    """
    Logistic Distribution using the following parameterization:

    f(x | loc, scale) = exp(-z) / (s * (1 + exp(-z))^2)
    
    where z = (x - loc) / scale

    Parameters
    ----------
    loc : float, positive
        Location parameter
    scale : float, positive
        Scale parameter

    Methods
    -------
    exp()
        Transforms self to LogLogistic

    Relationships
    -------------
    Let X be Logistic, a, b float. Then:
    * aX + b is Logistic
    * exp(X) is Log-Logistic
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

    def exp(self):
        return LogLogistic(alpha=exp(self.loc), beta=1/self.scale)

    # TODO: Gumbel - Gumbel = Logistic

class LogLogistic(distribution.Distribution):
    """
    LogLogistic Distribution using the following parameterization:

    f(x | a, b) = (b/a) * (x/a)^(b-1) / (1 + (x/a)^b)^2
    
    Parameters
    ----------
    alpha : float, positive
        Scale parameter
    beta : float, positive
        Shape parameter

    Methods
    -------
    log()
        Transforms self to Logistic

    Relationships
    -------------
    Let X be LogLogistic, k > 0 float. Then:
    * kX is LogLogistic
    * log(X) is Logistic
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
        assert alpha > 0, "alpha must be positive"
        assert beta > 0, "alpha must be positive"
        
        # Parameters
        self.alpha = alpha
        self.beta = beta

        # Scipy backend
        self.sp = fisk(c=beta, scale=alpha)

        super().__init__()

    def __repr__(self):
        return f"LogLogistic(alpha={self.alpha}, beta={self.beta})"

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return LogLogistic(other*self.alpha, self.beta)
        else:
            raise TypeError(f"Can't multiply objects of type {type(other)} by LogLogistic")

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1/other)
        else:
            raise TypeError(f"Can't divide objects of type {type(other)} by LogLogistic")

    def log(self):
        return Logistic(loc=np.log(self.alpha), scale=1/self.beta)
