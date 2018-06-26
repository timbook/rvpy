from . import binomial

class Bernoulli(binomial.Binomial):
    def __init__(self, p):
        super().__init__(n=1, p=p)

    def __repr__(self):
        return f"Bernoulli(p={self.p})"

    def to_binomial(self):
        return Binomial(n=1, p=self.p)
