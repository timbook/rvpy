class Distribution:
    def __init__(self):
        pass

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
        
