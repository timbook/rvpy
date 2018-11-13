class Distribution:
    """
    This is the base Distribution class from which all other univariate
    distributions inherit. Each subclass also calls the __init__ method herein,
    which gives the sublcasses their moments, median, and entropy.

    Methods
    -------
    pdf(x)
        If continuous, returns the pdf at point x
    pmf(x)
        If discrete, returns the pmf at point x
    cdf(x)
        Returns CDF at point x
    prob_interval(a, b)
        For a random variable X, returns P(a <= X < b). This is equivalent to
        cdf(b) - cdf(a)
    quantile(x)
        Returns the xth quantile of the random variable
    sample(*shape)
        Returns a random sampling of the random variable of the given shape
    """
    def __init__(self):
        self._moments()

    def _moments(self):
        m, v, s, k = self.sp.stats(moments = 'mvsk')
        self.mean = float(m)
        self.var = float(v)
        self.std = self.var**0.5
        self.skew = float(s)
        self.kurtosis = float(k)

        self.median = float(self.sp.median())
        self.entropy = float(self.sp.entropy())

    def __pos__(self):
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def pdf(self, x):
        return self.sp.pdf(x)

    def pmf(self, x):
        return self.sp.pmf(x)

    def cdf(self, x):
        return self.sp.cdf(x)

    def prob_interval(self, a, b):
        return self.cdf(b) - self.cdf(a)

    def quantile(self, x):
        return self.sp.ppf(x)

    def sample(self, *shape):
        return self.sp.rvs(size=shape)
