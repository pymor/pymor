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

## Examples

### Reduced Basis Method for Elliptic Problem

Loading a model.

```{code-cell}
from pymor.models.examples import thermal_block_example

fom_tb = thermal_block_example()
```

Check the model type.

```{code-cell}
print(fom_tb)
```

Show solution for a particular parameter value.

```{code-cell}
mu = [0.1, 0.2, 0.5, 1]
U = fom_tb.solve(mu)
fom_tb.visualize(U)
```

Run reduced basis method.

```{code-cell}
from pymor.algorithms.greedy import rb_greedy
from pymor.parameters.functionals import ExpressionParameterFunctional
from pymor.reductors.coercive import CoerciveRBReductor

parameter_space = fom_tb.parameters.space(0.0001, 1.)
reductor = CoerciveRBReductor(
    fom_tb,
    product=fom_tb.h1_0_semi_product,
    coercivity_estimator=ExpressionParameterFunctional('min(diffusion)',
                                                       fom_tb.parameters)
)
greedy_data = rb_greedy(fom_tb, reductor, parameter_space.sample_randomly(1000),
                        max_extensions=25)
rom_tb = greedy_data['rom']
```

Check ROM type.

```{code-cell}
print(rom_tb)
```

Show ROM reconstructed solution.

```{code-cell}
Urom = rom_tb.solve(mu)
Urec = reductor.reconstruct(Urom)
fom_tb.visualize((U, Urec))
```

Error in the solution.

```{code-cell}
fom_tb.visualize(U - Urec)
```

### Balanced Truncation for LTI System

Load model.

```{code-cell}
from pymor.models.examples import penzl_example

fom_lti = penzl_example()
```

Check model type.

```{code-cell}
print(fom_lti)
```

Show frequency response of the FOM.

```{code-cell}
w = (1e-1, 1e4)
_ = fom_lti.transfer_function.bode_plot(w)
```

Run balanced truncation.

```{code-cell}
from pymor.reductors.bt import BTReductor

bt = BTReductor(fom_lti)
rom_lti = bt.reduce(10)
```

Check ROM type.

```{code-cell}
print(rom_lti)
```

Show frequency response of the ROM.

```{code-cell}
_ = fom_lti.transfer_function.bode_plot(w)
_ = rom_lti.transfer_function.bode_plot(w, linestyle='dashed')
```

Error magnitude.

```{code-cell}
err_lti = fom_lti - rom_lti
_ = err_lti.transfer_function.mag_plot(w)
```

## Learning More

As a next step, you should read our {ref}`technical_overview` which discusses the
most important concepts and design decisions behind pyMOR. You can also follow our
growing set of {doc}`tutorials`, which focus on specific aspects of pyMOR.

Should you have any problems regarding pyMOR, questions or
[feature requests](<https://github.com/pymor/pymor/issues>), do not hesitate
to contact us via
[GitHub discussions](<https://github.com/pymor/pymor/discussions>)!
