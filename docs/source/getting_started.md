---
jupytext:
  text_representation:
   format_name: myst
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: ipynb,myst
    main_language: python
    text_representation:
      format_name: myst
      extension: .md
      format_version: '1.3'
      jupytext_version: 1.11.2
kernelspec:
  display_name: Python 3
  name: python3
---

```{try_on_binder}
```

```{code-cell}
:tags: [remove-cell]
:load: myst_code_init.py
```

(getting-started)=

# Getting Started

pyMOR includes a variety of model order reduction methods
for different types of models.
We illustrate how to use pyMOR on two basic examples and
give pointers to further documentation.

## Installation

pyMOR can be installed via `pip`. To follow this guide, pyMOR should be installed with GUI support:

```bash
pip install pymor[gui]
```

If you follow along in a [Jupyter](https://jupyter.org) notebook, you should install pyMOR using
the `jupyter` extra:

```bash
pip install pymor[jupyter]
```

We also provide [conda-forge](https://conda-forge.org) packages. 
For further details see [README.md](https://github.com/pymor/pymor/blob/main/README.md).

## Reduced Basis Method for Elliptic Problem

Here we show how to reduce a parametric linear elliptic problem using the Reduced Basis (RB) method.
As an example we use a standard 2x2 thermal-block problem,
which is given by the elliptic equation

$$
-\nabla \cdot [d(x, \mu) \nabla u(x, \mu)] = 1
$$

on the domain $[0, 1]^2$ with Dirichlet zero boundary values.
The domain is partitioned into 2x2 blocks and
the diffusion function $d(x, \mu)$ is constant
on each such block $i$ with value $\mu_i$.

```
(0,1)---------(1,1)
|        |        |
|  μ_2   |  μ_3   |
|        |        |
|------------------
|        |        |
|  μ_0   |  μ_1   |
|        |        |
(0,0)---------(1,0)
```

After discretization, the model has the form

$$
(L + \mu_0 L_0 + \mu_1 L_1 + \mu_2 L_2 + \mu_3 L_3) u(\mu) = f
$$

We can obtain this discrete full-order model in pyMOR using the
{func}`~pymor.models.examples.thermal_block_example` function
from the {mod}`pymor.models.examples` module:

```{code-cell}
from pymor.models.examples import thermal_block_example

fom_tb = thermal_block_example()
```

This uses pyMOR's {doc}`builtin discretization toolkit <tutorial_builtin_discretizer>`
to construct the model.
In the [builtin-discretizer tutorial](#tutorial-builtin-discretizer) you learn how
to define your own models using this toolkit.

`fom_tb` is an instance of {{StationaryModel}}, which encodes the mathematical structure
of the model through its {{Operators}}:

```{code-cell}
fom_tb
```

Its parameters can also be accessed:

```{code-cell}
fom_tb.parameters
```

Let us show the solution for particular parameter values:

```{code-cell}
mu = [0.1, 0.2, 0.5, 1]
U = fom_tb.solve(mu)
fom_tb.visualize(U)
```

To construct a reduced-order model using the reduced basis method,
we first need to create a {mod}`reductor <pymor.reductors>` which projects
the model onto a given reduced space:

```{code-cell}
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor

reductor = CoerciveRBReductor(
    fom_tb,
    product=fom_tb.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)',
                                                       fom_tb.parameters)
)
```

We use here {class}`~pymor.reductors.coercive.CoerciveRBReductor`, which
also assembles an a posterior error estimator for the error w.r.t. the
$H^1_0$-seminorm.
For that estimator, an parameter-dependent lower bound for the coercivity constant
of the problem is provided as an {class}`~pymor.parameters.functionals.ExpressionParameterFunctional`.

Using this reductor and its error estimator, we can build a reduced space with corrsponding
reduced-order model using a {func}`~pymor.algorithms.greedy.weak_greedy` algorithms:

```{code-cell}
from pymor.algorithms.greedy import rb_greedy

parameter_space = fom_tb.parameters.space(0.1, 1)
greedy_data = rb_greedy(fom_tb, reductor, parameter_space.sample_randomly(1000),
                        rtol=1e-2)
rom_tb = greedy_data['rom']
```

Here, the greedy algorithm operates on a randomly chosen set of 1000 training parameters from the
{{ParameterSpace}} of admissible parameter values returned by
{func}`~pymor.models.examples.thermal_block_example`.

We can see that the reduced-order model is also a `StationaryModel`,
but of lower order:

```{code-cell}
rom_tb
```

We can compute the reduced-order model's reconstructed solution
for the same parameter value and show the error:

```{code-cell}
Urom = rom_tb.solve(mu)
Urec = reductor.reconstruct(Urom)
fom_tb.visualize(U - Urec)
```

Finally, we check that the reduced-order model is indeed faster:

```{code-cell}
from time import perf_counter
tic = perf_counter()
fom_tb.solve(mu)
toc = perf_counter()
rom_tb.solve(mu)
tac = perf_counter()
print(f't_fom: {toc-tic}  t_rom: {tac-toc}  speedup: {(toc-tic)/(tac-toc)}')
```

## Balanced Truncation for LTI System

Here we consider a synthetic linear time-invariant (LTI) system of the form

$$
\begin{align*}
  \dot{x}(t) & = A x(t) + B u(t), \\
  y(t) & = C x(t),
\end{align*}
$$

where $x$ is the state, $u$ is the input, and $y$ is the output.

```{code-cell}
from pymor.models.examples import penzl_example

fom_lti = penzl_example()
```

The result is an {{LTIModel}}.

```{code-cell}
fom_lti
```

We can use the Bode plot to show the frequency response of the LTI system, i.e.,
to see which input frequencies are amplified and phase-shifted in the output.

```{code-cell}
w = (1e-1, 1e4)
_ = fom_lti.transfer_function.bode_plot(w)
```

We can run balanced truncation to obtain a reduced-order model.

```{code-cell}
from pymor.reductors.bt import BTReductor

bt = BTReductor(fom_lti)
rom_lti = bt.reduce(10)
```

The reduced-order model is again an {{LTIModel}}, but of lower order.

```{code-cell}
rom_lti
```

Looking at the error system, we can see which frequencies are well approximated.

```{code-cell}
err_lti = fom_lti - rom_lti
_ = err_lti.transfer_function.mag_plot(w)
```

## How to run demos?

While we consider pyMOR mainly as a library for building MOR applications, we
ship a few example scripts. These can be found in the `src/pymordemos`
directory of the source repository. Try launching one of them using the
`pymor-demo` script:

```
pymor-demo thermalblock --plot-err --plot-solutions 3 2 3 32
```

The demo scripts can also be launched directly from the source tree:

```
./thermalblock.py --plot-err --plot-solutions 3 2 3 32
```

This will reduce the so called thermal block problem using the reduced basis
method with a greedy basis generation algorithm. The thermal block problem
consists in solving the stationary heat equation

$$
\begin{align*}
  -\nabla \cdot [ d(x, \mu) \nabla u(x, \mu) ] & = 1,
  & \text{for } x \text{ in } \Omega, \\
  u(x, \mu) & = 0,
  & \text{for } x \text{ in } \partial\Omega,
\end{align*}
$$

on the domain $\Omega = (0, 1)^2$ for the unknown $u$.
The domain is partitioned into `XBLOCKS x YBLOCKS` blocks
(`XBLOCKS` and `YBLOCKS` are the first two arguments to `thermalblock.py`).
The thermal conductivity $d(x, \mu)$ is constant on each block $(i, j)$ with
value $\mu_{ij}$:


The real numbers $\mu_{ij}$ form the `XBLOCKS x YBLOCKS` - dimensional parameter
on which the solution depends.

Running `thermalblock.py` will first produce plots of two detailed
solutions of the problem for different randomly chosen parameters
using linear finite elements. (The size of the grid can be controlled
via the `--grid` parameter. The randomly chosen parameters will
actually be the same for each run, since a the random generator
is initialized with a fixed default seed in
{func}`~pymor.tools.random.default_random_state`.)

After closing the window, the reduced basis for model order reduction
is generated using a greedy search algorithm with error estimator.
The third parameter `SNAPSHOTS` of `thermalblock.py` determines how many
different values per parameter component $\mu_{ij}$ should be considered.
I.e. the parameter training set for basis generation will have the
size `SNAPSHOTS^(XBLOCKS x YBLOCKS)`. After the basis of size 32 (the
last parameter) has been computed, the quality of the obtained reduced model
(on the 32-dimensional reduced basis space) is evaluated by comparing the
solutions of the reduced and detailed models for new, randomly chosen
parameter values. Finally, plots of the detailed and reduced solutions, as well
as the difference between the two, are displayed for the random
parameter values which maximizes reduction error.

## Learning More

You can find more details on the above approaches,
and even more,
in the {doc}`tutorials`.
Specific questions are answered in
[GitHub discussions](https://github.com/pymor/pymor/discussions).
The {ref}`technical_overview` provides discussion on fundamental pyMOR concepts
and design decisions.

Should you have any problems regarding pyMOR, questions or
[feature requests](https://github.com/pymor/pymor/issues),
do not hesitate to contact us via
[GitHub discussions](https://github.com/pymor/pymor/discussions)
or [email](mailto:main.developers@pymor.org)!
