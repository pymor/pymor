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
# linear combination of non-parametric operators with parametric coefficients
print(repr(fom.operator))

# instantiate reductor that builds the ROM given some reduced basis;
# `product` is inner product w.r.t. which MOR error is estimated;
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

### Neural networks for parameter-dependent problems

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example()

# instantiate reductor with training and validation parameters and desired errors
from pymor.reductors.neural_network import NeuralNetworkReductor
reductor = NeuralNetworkReductor(fom,
                                 training_set=fom.parameters.space(0.1, 1).sample_uniformly(2),
                                 validation_set=fom.parameters.space(0.1, 1).sample_randomly(5),
                                 ann_mse=None, scale_outputs=True)
rom =  reductor.reduce(restarts=5)

# estimate and compute state-space MOR error
mu = rom.parameters.parse([0.1, 0.9, 0.2, 0.3])
u = rom.solve(mu)
print(f'Actual error: {(fom.solve(mu) - reductor.reconstruct(u)).norm(fom.h1_0_semi_product)}')
```

### Estimation of coercivity and continuity constants using the min/max-theta approaches

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import thermal_block_example
fom = thermal_block_example()

from pymor.parameters.functionals import MaxThetaParameterFunctional, MinThetaParameterFunctional
mu_bar = fom.parameters.parse([0.5, 0.5, 0.5, 0.5])
coercivity_estimator = MinThetaParameterFunctional(fom.operator.coefficients, mu_bar)
continuity_estimator = MaxThetaParameterFunctional(fom.operator.coefficients, mu_bar, gamma_mu_bar=0.5)

from pymor.parameters.functionals import ExpressionParameterFunctional
mu = fom.parameters.parse([0.1, 0.9, 0.2, 0.3])
naive_coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)
print(f"Naive coercivity constant estimate: {naive_coercivity_estimator.evaluate(mu)}")
print(f"Coercivity constant estimate using min-theta approach: {coercivity_estimator.evaluate(mu)}")
print(f"Continuity constant estimate using max-theta approach: {continuity_estimator.evaluate(mu)}")
```

### Estimation of coercivity constants using the successive constraints method

```{code-cell} ipython3
:tags: [remove-output]

from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg
problem = thermal_block_problem(num_blocks=(2, 2))
fom, _ = discretize_stationary_cg(problem, diameter=0.1)

from pymor.algorithms.scm import construct_scm_functionals
initial_parameter = problem.parameter_space.sample_randomly(1)[0]
training_set = problem.parameter_space.sample_randomly(50)
coercivity_estimator, _, _ = construct_scm_functionals(
            fom.operator, training_set, initial_parameter, product=fom.h1_0_semi_product, max_extensions=10, M=5)

from pymor.parameters.functionals import ExpressionParameterFunctional
mu = fom.parameters.parse([0.1, 0.9, 0.2, 0.3])
naive_coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)
print(f"Naive coercivity constant estimate: {naive_coercivity_estimator.evaluate(mu)}")
print(f"Coercivity constant estimate using successive constraints method: {coercivity_estimator.evaluate(mu)}")
```

### Parabolic problems using greedy algorithm

```{code-cell} ipython3

from pymor.models.examples import heat_equation_example
fom = heat_equation_example()
parameter_space = fom.parameters.space(1, 100)

from pymor.parameters.functionals import ExpressionParameterFunctional
coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
from pymor.reductors.parabolic import ParabolicRBReductor
reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)

from pymor.algorithms.greedy import rb_greedy
training_set = parameter_space.sample_uniformly(20)
greedy_data = rb_greedy(fom, reductor, training_set=parameter_space.sample_uniformly(20), max_extensions=10)
rom = greedy_data['rom']

# estimate and compute state-space MOR error
mu = rom.parameters.parse([10.])
u = rom.solve(mu)
print(f'Actual error: {(fom.solve(mu) - reductor.reconstruct(u)).norm(fom.h1_0_semi_product)}')
```

## LTI System MOR

Here we consider some of the methods for {{LTIModels}}.

```{code-cell} ipython3
:tags: [remove-output]

from pymor.models.examples import penzl_example
fom = penzl_example()

# balanced truncation
from pymor.reductors.bt import BTReductor
rom_bt = BTReductor(fom).reduce(10)

# iterative rational Krylov algorithm (IRKA)
from pymor.reductors.h2 import IRKAReductor
rom_irka = IRKAReductor(fom).reduce(10)
```

Download the code:
{download}`mor_methods.md`,
{nb-download}`mor_methods.ipynb`.
