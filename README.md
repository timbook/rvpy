# `rvpy` Python library working with random variables in an OOish way.

`rvpy` is a library for working with random variables in a "Pythonic" way.
Random variables (coming soon: random vectors) can be stored as objects to be
used in several common operations and conversions. `rvpy`'s distributional
methods (`pdf()`, `cdf()`, etc) are wrapped around `scipy.stats`. Relationships
between distributions (dunder methods, `to_*()`, etc) are bespoke.

```python
>>> import rvpy
>>> X = rvpy.Normal(3, 5)
>>> Z = rvpy.StandardNormal()
>>> X + Z
Normal(mu=3, sigma=5.0990195135927845)
>>> Z**2
ChiSq(df=1)
```

## Installation:
`rvpy` is available on PyPI and can easily be installed via

```bash
pip install rvpy
```
To install the version currently hosted here, simply clone and install locally:

```bash
git clone git@github.com:timbook/rvpy.git
cd rvpy
pip install .
```

## TODO:
* Various more distributions to add
    - Gumbel, Beta-Binoial, Gompertz, Logistic
* Move a fuller list of TODOs to a `TODO.md` file.
* A `CONTRIBUTING.md`

## Future ideas:
* `fit()` methods for each RV (choice of `"MLE"`, `"MOM"`?)
* Multivariate distributions
* Implementing a `.given()` method for conditional distributions
