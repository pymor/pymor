---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{try_on_binder}
```

```{code-cell} ipython3
:load: myst_code_init.py
:tags: [remove-cell]


```

# Available MOR methods

Here we give an overview over (most of) the available MOR methods implemented in pyMOR.
We provide short code snippets that show how to use these methods with pyMOR.
For more in-depth explanations we refer to the {doc}`tutorials`.

## Data approximation

### POD

```{code-cell} ipython3
:tags: [remove-output]

# generate some data to approximate
from pymor.models.examples import thermal_block_example
fom = thermal_block_example()
U = fom.solution_space.empty()
for mu in fom.parameters.space(0.1, 1).sample_randomly(10):
    U.append(fom.solve(mu))

# return first 3 POD modes and singular values
from pymor.algorithms.pod import pod
modes, singular_values = pod(U, modes=3)

# return modes with singular value larger than 1e-3
modes, _  = pod(U, atol=1e-3)

# return right-singular vectors
modes, _, coeffs = pod(U, return_reduced_coefficients=True)

# use slower but more accurate algorithm
# (default algorithm is only accurate up to half machine precision)
modes, _ = pod(U, method='qr_svd')
```

## Parametric MOR

Here we consider MOR methods for {{Models}} that depend on one or more {{Parameters}}.

### Reduced Basis method for parameter-separable, linear, coercive models

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example()

# FOM is parameter separable, i.e., system operator is a
# linear combination non-parametric operators
print(repr(fom.operator))

# instatiate reductor that builds the ROM given some reduced basis
# `product` is inner product w.r.t. which MOR error is estimated
# `coercivity_estimator` needs to return lower bound for the operator's
# coercivity constant (w.r.t. given `product`) for the given parameter values
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor
reductor = CoerciveRBReductor(
    fom,
    product=fom.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)

# note: use SimpleCoerciveRBReductor for faster offline phase but error estimator that
# only is accurate up to half machine precision

# use weak greedy algorithm to train the model
from pymor.algorithms.greedy import rb_greedy
greedy_data = rb_greedy(fom, reductor,
                        fom.parameters.space(0.1, 1).sample_randomly(1000),  # training set
                        rtol=1e-2)
rom = greedy_data['rom']

# estimate and compute state-space MOR error
mu = rom.parameters.parse([0.1, 0.9, 0.2, 0.3])
u = rom.solve(mu)
print(f'Error estimate: {rom.estimate_error(mu)}')
print(f'Actual error: {(fom.solve(mu) - reductor.reconstruct(u)).norm(fom.h1_0_semi_product)}')
```

Download the code:
{download}`mor_methods.md`,
{nb-download}`mor_methods.ipynb`.
