import numpy as np

class CUniform(Distribution):
    def __init__(self, a=0, b=1):
        assert b > a, "b must be larger than a"

        # Parameters
        self.a = a
        self.b = b

        # Moments
        self.mean = (a + b) / 2
        self.var = (b - a)**2 / 12
        self.std = self.var**0.5
        self.skew = 0
        self.kurtosis = -6/5

    def __repr__(self):
        return f"CUniform(a={self.a}, b={self.b})"

    def __pos__(self):
        return self

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


    def sample(self, *shape):
        return (self.b - self.a) * np.random.rand(*shape) + self.a

    def pdf(self, x):
        return np.where((x >= self.a) & (x <= self.b), 1 / (self.b - self.a), 0)

    def cdf(self, x):
        above_a = (x >= self.a) * 1
        return np.where(x < self.b, (x-self.a) / (self.b-self.a), 1) * above_a

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
    
