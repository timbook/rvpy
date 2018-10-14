import numpy as np
from scipy.stats import weibull_min, rayleigh
from . import distribution
from . import gamma as gamm

class Weibull(distribution.Distribution):
    def __init__(self, gamma, beta):
        assert gamma > 0 and beta > 0, "gamma and beta must be positive"

        self.gamma = gamma
        self.beta = beta

        # Scipy backend
        self.sp = weibull_min(c=gamma, scale=beta)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Weibull(gamma={self.gamma}, beta={self.beta})"

    def to_exponential(self):
        assert self.gamma == 1, "gamma must be 1 to cast as Exponential"
        return gamm.Exponential(self.beta)

    def to_rayleigh(self):
        assert self.gamma == 2, "beta must be 2 to cast as Rayleigh"
        return Rayleigh(self.beta / 2**0.5)

    # TODO: .to_gumbel() --> ???

class Rayleigh(Weibull):
    def __init__(self, scale):
        assert scale > 0, "scale parameter must be positive"

        self.scale = scale
        self.sigma = scale

        # Scipy backend
        self.sp = rayleigh(scale=scale)

        # Initialize super
        super().__init__(2, scale * 2**0.5)

    def __repr__(self):
        return f"Rayleigh(scale={self.scale})"

    def to_weibull(self):
        return Weibull(2, self.scale * 2**0.5)

    # def __pow__(self, k):
        # TODO: Rayleigh(s)**2 --> Gamma(1, 2s**2)
        # TODO: Rayleigh(1)**2 --> ChiSq(2)

