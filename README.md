# `rvpy` Python library working with random variables in an OOish way.

`rvpy` is a library for working with random variables in a "Pythonic" way.
Random variables (coming soon: random vectors) can be stored as objects to be
used in several common operations and conversions.

```python
>>> import rvpy
>>> X = rvpy.Normal(3, 5)
>>> Z = rvpy.StandardNormal()
>>> X + Z
Normal(mu=3, sigma=5.0990195135927845)
>>> Z**2
ChiSq(df=1)
```

## TODO:
* Documentation
* setup.py
* `fit()` methods for each RV (`"MLE"`, `"MOM"`)
* Remaining continuous distributions to add:
    - Weibull
    - Rayleigh
    - Double exponential / Laplace
    - Cauchy
    - Lognormal
* Remaining discrete distributions to add:
    - DUniform
    - Hypergeometric
    - Poisson
    - Beta-binomial
    - Negative binomial
    - Geometric


