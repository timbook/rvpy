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
        try:
            return self.sp.pdf(x)
        except AttributeError:
            print("This random variable has no pdf method!")

    def pmf(self, x):
        try:
            return self.sp.pmf(x)
        except AttributeError:
            print("This random variable has no pmf method!")

    def cdf(self, x):
        try:
            return self.sp.cdf(x)
        except AttributeError:
            print("This random variable has no pdf method!")

    def quantile(self, x):
        try:
            return self.sp.ppf(x)
        except AttributeError:
            print("This random variable has no quantile method!")

    def sample(self, *shape):
        try:
            return self.sp.rvs(size=shape)
        except AttributeError:
            print("Cannot sample fromthis distribution!")
        
