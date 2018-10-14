import numpy as np
from scipy.stats import randint
from . import distribution

class DUniform(distribution.Distribution):
    def __init__(self, a, b):
        assert isinstance(a, int) and isinstance(b, int), \
                "DUniform bounds must be integers"
        assert a < b, "a must be less than b"

        # Parameters
        self.a = a
        self.b = b

        # Scipy backend
        self.sp = randint(a, b + 1)

        # Initalize super
        super().__init__()

    def __repr__(self):
        return f"DUniform(a={self.a}, b={self.b})"

    def __add__(self, c):
        assert isinstance(c, int), \
                "Only adding integers to DUniform is supported"

        return DUniform(self.a + c, self.b + c)

    # TODO: __neg__?
