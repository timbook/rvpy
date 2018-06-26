class Distribution:
    def __init__(self):
        pass

    def _moments(self):
        m, v, s, k = self.sp.stats(moments = 'mvsk')
        self.mean = float(m)
        self.var = float(v)
        self.std = v**0.5
        self.skew = float(s)
        self.kurtosis = float(k)

        self.median = float(self.sp.median())
        self.entropy = float(self.sp.entropy())
    

    def __pos__(self):
        return self

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
        
