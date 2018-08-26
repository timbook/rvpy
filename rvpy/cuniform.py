import numpy as np
from scipy.stats import uniform
from . import distribution

class CUniform(distribution.Distribution):
    def __init__(self, a=0, b=1):
        assert b > a, "b must be larger than a"

        # Parameters
        self.a = a
        self.b = b

        # Scipy backend
        self.sp = uniform(a, b - a)

        # Intialize super
        super().__init__()

    def __repr__(self):
        return f"CUniform(a={self.a}, b={self.b})"

    def __add__(self, c):
        if isinstance(c, (int, float)):
            return CUniform(self.a + c, self.b + c)
        else:
            raise TypeError("Only scalar addition for CUniforms is supported.")

    def __radd__(self, c):
        return self.__add__(c)

    def __mul__(self, c):
        if isinstance(c, (int, float)):
            return CUniform(self.a * c, self.b * c)
        else:
            raise TypeError("Only scalar multiplication for CUniforms is supported.")

    def __rmul__(self, c):
        return self.__mul__(c)

    def mgf(self, t):
        # TODO: Why is t = 0 throwing a warning?
        def mgf_not_0(t):
            num = np.exp(t*self.b) - np.exp(t*self.a)
            den = t * (self.b - self.a)
            return num / den
        return np.where(t == 0, 1, mgf_not_0(t))

    # TODO: -logU = Exp(1)?
    # TODO: U**n = Beta(1/n, 1)
    # TODO: U.to_beta() --> Beta(1, 1)
    
