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

# Tutorial: Interpolation-based methods for LTI systems

Here we discuss interpolation-based methods
(aka. moment matching, aka. Krylov subspace methods)
for LTI systems,
and demonstrate it on the heat equation example from
{doc}`tutorial_lti_systems`.
We start with simpler approaches (interpolation at infinity and at zero) and
then move on to bitangential Hermite interpolation
which is directly supported in pyMOR.

First, we import necessary packages.

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel

plt.rcParams['axes.grid'] = True
```

Then we build the matrices

```{code-cell}
:load: heat_equation.py


```

and form the full-order model.

```{code-cell}
fom = LTIModel.from_matrices(A, B, C, E=E)
```

## Interpolation at infinity

Given an LTI system

```{math}
\begin{align}
    \dot{x}(t) & = A x(t) + B u(t), \\
    y(t) & = C x(t) + D u(t),
\end{align}
```

the most straightforward interpolation method is using a Krylov subspace

```{math}
V =
\begin{bmatrix}
    B & A B & \cdots & A^{k - 1} B
\end{bmatrix}
```

to perform a Galerkin projection.
This will achieve interpolation of the first $k$ moments at infinity
of the transfer function.

Download the code:
{download}`tutorial_interpolation.md`,
{nb-download}`tutorial_interpolation.ipynb`.
