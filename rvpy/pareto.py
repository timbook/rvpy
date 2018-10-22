import numpy as np
from scipy.stats import pareto
from . import distribution

class Pareto(distribution.Distribution):
    def __init__(self, alpha, beta):
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
