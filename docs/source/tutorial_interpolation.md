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
and demonstrate it on
[Penzl' FOM example](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Penzl%27s_FOM).

```{code-cell}
from pymor.models.examples import penzl_example

fom = penzl_example()
```

We start with simpler approaches (interpolation at infinity and at zero) and
then move on to bitangential Hermite interpolation
which is directly supported in pyMOR.

## Interpolation at infinity

Given an LTI system

```{math}
\begin{align}
  \dot{x}(t) & = A x(t) + B u(t), \\
  y(t) & = C x(t) + D u(t),
\end{align}
```

the most straightforward interpolation method is using Krylov subspaces

```{math}
\newcommand{\T}{\operatorname{T}}
\begin{align*}
  V
  & =
  \begin{bmatrix}
    B & A B & \cdots & A^{r - 1} B
  \end{bmatrix}, \\
  W
  & =
  \begin{bmatrix}
    C^{\T} & A^{\T} C^{\T} & \cdots & {\left(A^{\T}\right)}^{r - 1} C^{\T}
  \end{bmatrix}
\end{align*}
```

to perform a Galerkin projection.
This will achieve interpolation of the first $2 r$ moments at infinity
of the transfer function.
The moments at infinity, also called *Markov parameters*,
are the coefficients in the Laurent series expansion:

```{math}
\begin{align*}
  H(s)
  & = C {(s I - A)}^{-1} B + D \\
  & = C \frac{1}{s} {\left(I - \frac{1}{s} A \right)}^{-1} B + D \\
  & = C \frac{1}{s}
    \sum_{k = 0}^{\infty} {\left(\frac{1}{s} A \right)}^{k}
    B + D \\
  & = D + \sum_{k = 0}^{\infty} C A^k B \frac{1}{s^{k + 1}} \\
  & = D + \sum_{k = 1}^{\infty} C A^{k - 1} B \frac{1}{s^{k}}.
\end{align*}
```

The moments at infinity are thus

```{math}
M_0(\infty) = D, \quad
M_k(\infty) = C A^{k - 1} B,  \text{ for } k \ge 1.
```

```{code-cell}
from pymor.algorithms.krylov import arnoldi

r_max = 5
V = arnoldi(fom.A, fom.E, fom.B.as_range_array(), r_max)
W = arnoldi(fom.A.H, fom.E.H, fom.C.as_source_array(), r_max)
```

Projection gives

```{code-cell}
from pymor.reductors.basic import LTIPGReductor

pg = LTIPGReductor(fom, W, V)
roms = [pg.reduce(i + 1) for i in range(r_max)]
```

Plotting magnitude plot.

```{code-cell}
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

w = (1e-1, 1e4)

fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
for i, rom in enumerate(roms):
    rom.transfer_function.mag_plot(w, ax=ax, label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

Plotting error.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    err = fom - rom
    err.transfer_function.mag_plot(w, ax=ax, label=f'Error $r = {i + 1}$')
_ = ax.legend()
```

Poles of the ROMs.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    poles = rom.poles()
    ax.plot(poles.real, poles.imag, '.', label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

## Interpolation at zero

The next option is using an inverse Krylov subspace.

```{math}
V =
\begin{bmatrix}
  A^{-1} B & A^{-2} B & \cdots & A^{-k} B
\end{bmatrix}
```

This will achieve interpolation of the first $k$ moments at zero of the transfer
function.

```{code-cell}
from pymor.operators.constructions import InverseOperator

r_max = 5
v0 = fom.A.apply_inverse(fom.B.as_range_array())
V = arnoldi(InverseOperator(fom.A), InverseOperator(fom.E), v0, r_max)
w0 = fom.A.apply_inverse_adjoint(fom.C.as_source_array())
W = arnoldi(InverseOperator(fom.A.H), InverseOperator(fom.E.H), w0, r_max)
```

Projection gives

```{code-cell}
pg = LTIPGReductor(fom, W, V)
roms = [pg.reduce(i + 1) for i in range(r_max)]
```

Plotting magnitude plot.

```{code-cell}
fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
for i, rom in enumerate(roms):
    rom.transfer_function.mag_plot(w, ax=ax, label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

Plotting error.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    err = fom - rom
    err.transfer_function.mag_plot(w, ax=ax, label=f'Error $r = {i + 1}$')
_ = ax.legend()
```

Poles of the ROMs.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    poles = rom.poles()
    ax.plot(poles.real, poles.imag, '.', label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

## Interpolation at arbitrary, finite point

More general approach is a rational Krylov subspace.

```{math}
V =
\begin{bmatrix}
  (s I - A)^{-1} B & (s I - A)^{-2} B & \cdots & (s I - A)^{-k} B
\end{bmatrix}
```

This will achieve interpolation of the first $k$ moments at {math}`s` of the
transfer function.

```{code-cell}
from pymor.algorithms.gram_schmidt import gram_schmidt

s = 200j
r_max = 5
v0 = (s * fom.E - fom.A).apply_inverse(fom.B.as_range_array())
V_complex = arnoldi(InverseOperator(s * fom.E - fom.A), InverseOperator(fom.E), v0, r_max)
V = fom.A.source.empty(reserve=2 * r_max)
for v1, v2 in zip(V_complex.real, V_complex.imag):
  V.append(v1)
  V.append(v2)
_ = gram_schmidt(V, atol=0, rtol=0, copy=False)
w0 = (s * fom.E - fom.A).apply_inverse_adjoint(fom.C.as_source_array())
W_complex = arnoldi(InverseOperator(s * fom.E.H - fom.A.H), InverseOperator(fom.E), w0, r_max)
W = fom.A.source.empty(reserve=2 * r_max)
for w1, w2 in zip(W_complex.real, W_complex.imag):
  W.append(w1)
  W.append(w2)
_ = gram_schmidt(W, atol=0, rtol=0, copy=False)
```

Projection gives

```{code-cell}
pg = LTIPGReductor(fom, W, V)
roms = [pg.reduce(2 * (i + 1)) for i in range(r_max)]
```

Plotting magnitude plot.

```{code-cell}
fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
for i, rom in enumerate(roms):
    r = 2 * (i + 1)
    rom.transfer_function.mag_plot(w, ax=ax, label=f'ROM $r = {r}$')
_ = ax.legend()
```

Plotting error.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    r = 2 * (i + 1)
    err = fom - rom
    err.transfer_function.mag_plot(w, ax=ax, label=f'Error $r = {r}$')
_ = ax.legend()
```

Poles of the ROMs.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    r = 2 * (i + 1)
    poles = rom.poles()
    ax.plot(poles.real, poles.imag, '.', label=f'ROM $r = {r}$')
_ = ax.legend()
```

Download the code:
{download}`tutorial_interpolation.md`,
{nb-download}`tutorial_interpolation.ipynb`.
