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
:load: myst_code_init.py
:tags: [remove-cell]
```

# How to run demos?

While we consider pyMOR mainly as a library for building MOR applications, we
ship a few example scripts. These can be found in the `src/pymordemos`
directory of the source repository (some are available as Jupyter notebooks in
the `notebooks` directory). Try launching one of them using the `pymor-demo`
script:

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

```
- ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1     for x in Ω
                  u(x, μ)   = 0     for x in ∂Ω
```

on the domain Ω = [0,1]^2 for the unknown u. The domain is partitioned into
`XBLOCKS x YBLOCKS` blocks (`XBLOCKS` and `YBLOCKS` are the first
two arguments to `thermalblock.py`). The thermal conductivity d(x, μ)
is constant on each block (i,j) with value μ_ij:

```
(0,1)------------------(1,1)
|        |        |        |
|  μ_11  |  μ_12  |  μ_13  |
|        |        |        |
|---------------------------
|        |        |        |
|  μ_21  |  μ_22  |  μ_23  |
|        |        |        |
(0,0)------------------(1,0)
```

The real numbers μ_ij form the `XBLOCKS x YBLOCKS` - dimensional parameter
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
different values per parameter component μ_ij should be considered.
I.e. the parameter training set for basis generation will have the
size `SNAPSHOTS^(XBLOCKS x YBLOCKS)`. After the basis of size 32 (the
last parameter) has been computed, the quality of the obtained reduced model
(on the 32-dimensional reduced basis space) is evaluated by comparing the
solutions of the reduced and detailed models for new, randomly chosen
parameter values. Finally, plots of the detailed and reduced solutions, as well
as the difference between the two, are displayed for the random
parameter values which maximises reduction error.
