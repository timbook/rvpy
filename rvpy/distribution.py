class Distribution:
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
        
