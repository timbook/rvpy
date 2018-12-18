# TODO List

## 0.4 (Next update)
* Multivariate distributions

## 0.5 (The "Bayes Update")
* Goal is to add a `.given()` method for certain classes for conditional distributions.
    - e.g. for `X`, `Y` Normal, `X.given(mu=Y)` would yield another Normal.
    - Potentially incorporate `__or__`, so the above syntax could be written `X | {'mu': Y}`

Achieving the following wish list would probably bring this to 1.0:
## Future Releases
* A `CONTRIBUTING.md` file
* A code of conduct
* A `.fit()` method on an "empty" distribution that would yield empirical parameters and moments (potentially with `MLE` and `MOM` options).
