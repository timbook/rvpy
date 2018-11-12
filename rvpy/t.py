import numpy as np
from scipy.stats import t
from . import distribution
from . import f, cauchy

class T(distribution.Distribution):
    """
    Student's T Distribution

    Parameters
    ----------
    df : float, positive (usually integer)
        Degrees of freedom

    Methods
    -------
    to_cauchy()
        Converts to StandardCauchy if df == 1

    Relationships
    -------------
    Let X be T-distributed
    * X**2 is F-distributed
    * 1 / X**2 is F-distributed
    """
    def __init__(self, df):
        """
        Parameters
        ----------
        df : float, positive (usually integer)
            Degrees of freedom
        """
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
        assert k == 2 or k == -2, "Only squaring t distribution is supported"
        if k == 2:
            return f.F(1, self.df)
        elif k == -2:
            return f.F(self.df, 1)

    def to_cauchy(self):
        assert self.df == 1, "Can only convert to Cauchy if df == 1"
        return cauchy.StandardCauchy()


    # TODO: Fix moments from low df
    # TODO: to_noncentral_t()
    # TODO: Throw error with .mgf()
