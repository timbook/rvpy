import numpy as np
from scipy.stats import beta as bta
from . import distribution
from . import cuniform

class Beta(distribution.Distribution):
    def __init__(self, alpha, beta):
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


