# `rvpy` Python library working with random variables in an OOish way.

`rvpy` is a library for working with random variables in a "Pythonic" way.
Random variables (coming soon: random vectors) can be stored as objects to be
used in several common operations and conversions. `rvpy`'s distributional
methods (`pdf()`, `cdf()`, etc) are wrapped around `scipy.stats`. Relationships
between distributions (dunder methods, `to_*()`) are bespoke.

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
* Various more distributions to add
* Some graph or master resource for discovering relationships between distributions

## Future ideas:
* `fit()` methods for each RV (choice of `"MLE"`, `"MOM"`?)
* Multivariate distributions
* Maybe implementing a `.given()` method for conditional distributions

