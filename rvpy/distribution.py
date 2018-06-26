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
            print("This random variable has no pdf method!")

    def cdf(self, x):
        try:
            return self.sp.cdf(x)
        except AttributeError:
            print("This random variable has no pdf method!")

