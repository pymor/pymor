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
We illustrate it here on two examples and
give pointers to further documentation.

## Reduced Basis Method for Elliptic Problem

Here we use a 2x2 thermal block example,
which is described by the elliptic equation

$$
-\nabla \cdot [d(x, \mu) \nabla u(x, \mu)] = 1
$$

on the domain $[0, 1]^2$ with Dirichlet zero boundary values.
The domain is partitioned into 2x2 blocks and
the diffusion function $d(x, \mu)$ is constant
on each such block $i$ with value $\mu_i$.
After discretization, the model has the form

$$
(L + \mu_0 L_0 + \mu_1 L_1 + \mu_2 L_2 + \mu_3 L_3) u(\mu) = f
$$

```{code-cell}
from pymor.models.examples import thermal_block_example

fom_tb = thermal_block_example()
```

We can check that the result is indeed a {{StationaryModel}}.

```{code-cell}
fom_tb
```

Its parameters can also be accessed.

```{code-cell}
fom_tb.parameters
```

Let us show the solution for a particular parameter value.

```{code-cell}
mu = [0.1, 0.2, 0.5, 1]
U = fom_tb.solve(mu)
fom_tb.visualize(U)
```

We can construct a reduced-order model using a reduced basis method.

```{code-cell}
from pymor.algorithms.greedy import rb_greedy
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor

parameter_space = fom_tb.parameters.space(0.1, 1.)
reductor = CoerciveRBReductor(
    fom_tb,
    product=fom_tb.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)',
                                                       fom_tb.parameters)
)
greedy_data = rb_greedy(fom_tb, reductor, parameter_space.sample_randomly(1000),
                        rtol=1e-5)
rom_tb = greedy_data['rom']
```

We can see that the reduced-order model is also a `StationaryModel`,
but of lower order.

```{code-cell}
rom_tb
```

We can compute the reduced-order model's reconstructed solution
for the same parameter value and show the error.

```{code-cell}
Urom = rom_tb.solve(mu)
Urec = reductor.reconstruct(Urom)
fom_tb.visualize(U - Urec)
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

## Learning More

You can find more details on the above approaches,
and even more,
in the {doc}`tutorials`.
Specific questions are answered in the {doc}`howtos` and
[GitHub discussions](https://github.com/pymor/pymor/discussions).
The {ref}`technical_overview` provides discussion on fundamental pyMOR concepts
and design decisions.

Should you have any problems regarding pyMOR, questions or
[feature requests](https://github.com/pymor/pymor/issues),
do not hesitate to contact us via
[GitHub discussions](https://github.com/pymor/pymor/discussions)
or [email](mailto:main.developers@pymor.org)!
