import numpy as np
from scipy.stats import t
from . import distribution

class TDist(distribution.Distribution):
    def __init__(self, df):
        assert df > 0, "df must be positive"

        # Parameters
        self.df = df

        # Scipy backend
        self.sp = t(df)

        # Initialize super - does nothing yet.
        super().__init__()

    def __repr__(self):
        return f"TDist(df={self.df})"

    # TODO: to_cauchy() if df == 1
    # TODO: to_noncentral_t()
    # TODO: X**2 = F(1, df)
