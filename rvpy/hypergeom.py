import numpy as np
from scipy.stats import hypergeom
from . import distribution

class Hypergeometric(distribution.Distribution):
    def __init__(self, N, M, K):
        assert N >= 0 and M >= 0 and K >= 0, \
                "All parameters of hypergeometric distribution must be nonnegative"
        assert K < N and M < N, "K and M must be less than N"

        # Parameters
        self.N = N
        self.M = M
        self.K = K

        # Scipy backend
        self.sp = hypergeom(M=N, n=M, N=K)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"Hypergeometric(N={self.N}, M={self.M}, K={self.K})"
