import numpy as np
from scipy.stats import gamma
from . import distribution

class Gamma(distribution.Distribution):
    def __init__(self, alpha, beta):
        assert alpha > 0 and beta > 0, "alpha and beta must be positive"

        self.alpha = alpha
        self.beta = beta

        # Scipy backend
        self.sp = gamma(a=alpha, scale=beta)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Gamma(alpha={self.alpha}, beta={self.beta})"

    def __add__(self, other):
        if isinstance(other, Gamma) and (self.beta == other.beta):
            if self.beta != other.beta:
                raise ValueError("Scale paramters of Gamma families must match")
            else:
                return Gamma(self.alpha + other.alpha, self.beta)
        else:
            raise TypeError("Only addition of Gamma families supported")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Gamma(self.alpha, self.beta / other)
        else:
            raise TypeError("Only multiplication by scalar supported")

    def __truediv__(self, other):
        if isinstance(other, (int, float)) and other != 0:
            return self.__mul__(1 / other)
        else:
            raise ZeroDivisionError("Cannot divide by zero!")

    def mgf(self, t):
        return np.where(t < 1/self.beta,
                   (1 - self.beta * t) ** (-self.alpha),
                   np.nan
               )

    def to_exponential(self):
        assert self.alpha == 1, "Alpha must be 1 to downcast"
        return Exponential(self.beta)

class Exponential(Gamma):
    def __init__(self, rate):
        # Get Gamma distribution initialization
        super().__init__(1, 1/rate)

        # Parameters
        self.rate = rate
        self.scale = 1 / rate

    def __repr__(self):
        return f"Exponential(rate={self.rate})"

    def to_gamma(self):
        return Gamma(alpha=1, beta=self.scale)

class ChiSq(Gamma):
    def __init__(self, df):
        # Get Gamma distribution initialization
        super().__init__(alpha=2*df, beta=2)

        # Parameters
        self.df = df

    def __repr__(self):
        return f"ChiSq(df={self.df})"

    def __add__(self, other):
        if isinstance(other, ChiSq):
            return ChiSq(self.df + other.df)
        else:
            return self.to_gamma() + other

    def to_gamma(self):
        return Gamma(alpha=2*df, beta=2)
