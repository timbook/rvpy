import numpy as np
from scipy.stats import t
from . import distribution
from . import f

class T(distribution.Distribution):
    def __init__(self, df):
        assert df > 0, "df must be positive"

        # Parameters
        self.df = df

        # Scipy backend
        self.sp = t(df)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"T(df={self.df})"

    def __pow__(self, k):
        assert k == 2, "Only squaring t distribution is supported"
        return f.F(1, self.df)

    # TODO: Fix moments from low df
    # TODO: to_cauchy() if df == 1
    # TODO: to_noncentral_t()
    # TODO: Throw error with .mgf()
