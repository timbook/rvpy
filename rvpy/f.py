import numpy as np
from scipy.stats import f
from . import distribution
from . import t

class F(distribution.Distribution):
    """
    Snedecor's F Distribution

    Parameters
    ----------
    df1 : float, positive
        Numerator degrees of freedom
    df2 : float, positive
        Denominator degrees of freedom

    Methods
    -------
    None

    Relationships
    -------------
    None implemented
    """
    def __init__(self, df1, df2):
        """
        Parameters
        ----------
        df1 : float, positive
            Numerator degrees of freedom
        df2 : float, positive
            Denominator degrees of freedom
        """
        assert df1 > 0 and df2 > 0, "degrees of freedom must be positive"

        # Parameters
        self.df1 = df1
        self.df2 = df2

        # Scipy backend
        self.sp = f(df1, df2)

        # Initialize super
        super().__init__()

    def __repr__(self):
        return f"F(df1={self.df1}, df2={self.df2})"
