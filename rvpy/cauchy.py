import numpy as np
from scipy.stats import cauchy
from . import distribution
from . import t

class Cauchy(distribution.Distribution):
    def __init__(self, loc=0, scale=1):
        assert scale > 0, "scale parameter must be positive"

        # Parameters
        self.loc = loc
        self.scale = scale

        # Scipy backend
        self.sp = cauchy(loc, scale)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Cauchy(loc={self.loc}, scale={self.scale})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Cauchy(self.loc + other, self.scale)
        elif isinstance(other, Cauchy):
            return Cauchy(self.loc + other.loc, self.scale + other.scale)
        else:
            raise TypeError(f"Can't add objects of type {type(other)} to Cauchy")

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return self.__add__(-other)
        elif isinstance(other, Cauchy):
            return Cauchy(self.loc - other.loc, self.scale + other.scale)
        else:
            raise TypeError(f"Can't subtract objects of type {type(other)} from Cauchy")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Cauchy(other*self.loc, abs(other)*self.scale)
        else:
            raise TypeError(f"Can't multiply objects of type {type(other)} to Cauchy")

    def __truediv__(self, other):
        return self.__mul__(1/other)

    def __rtruediv__(self, other):
        assert self.loc == 0, 'Can only invert Cauchy distributions with location 0'
        if other == 1:
            return Cauchy(0, 1/self.scale)
        elif isinstance(other, (int, float)):
            return other*Cauchy(0, 1/self.scale)
        else:
            raise TypeError(f"Can't divide objects of type {type(other)} by Cauchy")

    def to_standard(self):
        assert self.loc == 0 and self.scale == 1, \
                "Must have Cauchy(0, 1) to convert to StandardCauchy"
        return StandardCauchy()

    def to_t(self):
        return self.to_standard().to_t()

class StandardCauchy(Cauchy):
    def __init__(self):
        super().__init__(0, 1)

    def __repr__(self):
        return f"StandardCauchy(loc=0, scale=1)"

    def to_nonstandard(self):
        return Cauchy(0, 1)

    def to_t(self):
        return t.T(1)


