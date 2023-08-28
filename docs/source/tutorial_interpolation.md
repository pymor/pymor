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
(also known as:
moment matching,
Krylov subspace methods, and
Padé approximation)
for LTI systems,
and demonstrate it on
[Penzl' FOM example](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Penzl%27s_FOM).

```{code-cell}
from pymor.models.examples import penzl_example

fom = penzl_example()
```

We start with simpler approaches
(higher-order interpolation at single point {cite}`G97`) and
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

to perform a (Petrov-)Galerkin projection.
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

We can compute {math}`V` and {math}`W` using the
{func}`~pymor.algorithms.krylov.arnoldi` function.

```{code-cell}
from pymor.algorithms.krylov import arnoldi

r_max = 5
V = arnoldi(fom.A, fom.E, fom.B.as_range_array(), r_max)
W = arnoldi(fom.A.H, fom.E.H, fom.C.as_source_array(), r_max)
```

We then use those to perform a Petrov-Galerkin projection gives

```{code-cell}
from pymor.reductors.basic import LTIPGReductor

pg = LTIPGReductor(fom, W, V)
roms = [pg.reduce(i + 1) for i in range(r_max)]
```

To compare, we first draw magnitude plots.

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

As expected, we see good approximation for higher frequencies.
Drawing the magnitude of the error makes it clearer.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    err = fom - rom
    err.transfer_function.mag_plot(w, ax=ax, label=f'Error $r = {i + 1}$')
_ = ax.legend()
```

To check the stability of the ROM's, we can plot their poles.

```{code-cell}
fig, ax = plt.subplots()
markers = '.x+12'
for (i, rom), marker in zip(enumerate(roms), markers):
    poles = rom.poles()
    ax.plot(poles.real, poles.imag, marker, label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

We see that they are all asymptotically stable.

## Interpolation at zero

The next option is using an inverse Krylov subspaces
(more commonly called the Padé approximation).

```{math}
\begin{align*}
  V
  & =
  \begin{bmatrix}
    A^{-1} B & A^{-2} B & \cdots & A^{-r} B
  \end{bmatrix}, \\
  W
  & =
  \begin{bmatrix}
    A^{-\T} C^{\T}
    & {\left(A^{-\T}\right)}^{2} C^{\T}
    & \cdots
    & {\left(A^{-\T}\right)}^{r} C^{\T}
  \end{bmatrix}
\end{align*}
```

This will achieve interpolation of the first $2 r$ moments at zero of the
transfer function.

```{math}
\begin{align*}
  H(s)
  & = C {(s I - A)}^{-1} B + D \\
  & = C {\left(A \left(s A^{-1} - I\right)\right)}^{-1} B + D \\
  & = C {\left(s A^{-1} - I\right)}^{-1} A^{-1} B + D \\
  & = -C {\left(I - s A^{-1}\right)}^{-1} A^{-1} B + D \\
  & = -C
    \sum_{k = 0}^{\infty} {\left(s A^{-1} \right)}^{k}
    A^{-1} B + D \\
  & = D - \sum_{k = 0}^{\infty} C A^{-(k + 1)} B s^k.
\end{align*}
```

The moments at zero are

```{math}
M_0(0) = D - C A^{-1} B, \quad
M_k(0) = -C A^{-(k + 1)} B,  \text{ for } k \ge 1.
```

We can again use the {func}`~pymor.algorithms.krylov.arnoldi` function
to compute {math}`V` and {math}`W`.

```{code-cell}
from pymor.operators.constructions import InverseOperator

r_max = 5
v0 = fom.A.apply_inverse(fom.B.as_range_array())
V = arnoldi(InverseOperator(fom.A), InverseOperator(fom.E), v0, r_max)
w0 = fom.A.apply_inverse_adjoint(fom.C.as_source_array())
W = arnoldi(InverseOperator(fom.A.H), InverseOperator(fom.E.H), w0, r_max)
```

Then, in the same way, compute the Petrov-Galerkin projection...

```{code-cell}
pg = LTIPGReductor(fom, W, V)
roms = [pg.reduce(i + 1) for i in range(r_max)]
```

...and draw the magnitude plot.

```{code-cell}
fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
for i, rom in enumerate(roms):
    rom.transfer_function.mag_plot(w, ax=ax, label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

Now, because of interpolation at zero, we get good approximation for lower
frequencies.
The error plot shows this better.

```{code-cell}
fig, ax = plt.subplots()
for i, rom in enumerate(roms):
    err = fom - rom
    err.transfer_function.mag_plot(w, ax=ax, label=f'Error $r = {i + 1}$')
_ = ax.legend()
```

Checking stability using the poles of the ROMs.

```{code-cell}
fig, ax = plt.subplots()
for (i, rom), marker in zip(enumerate(roms), markers):
    poles = rom.poles()
    ax.plot(poles.real, poles.imag, marker, label=f'ROM $r = {i + 1}$')
_ = ax.legend()
```

## Interpolation at an arbitrary finite point

More general approach is a rational Krylov subspace
(also known as the shifted Padé approximation).

```{math}
\newcommand{\H}{\operatorname{H}}
\begin{align*}
  V
  & =
  \begin{bmatrix}
    (\sigma I - A)^{-1} B
    & (\sigma I - A)^{-2} B
    & \cdots
    & (\sigma I - A)^{-r} B
  \end{bmatrix}, \\
  W
  & =
  \begin{bmatrix}
    (\sigma I - A)^{-\H} C^{\T}
    & {\left((\sigma I - A)^{-\H}\right)}^{2} C^{\T}
    & \cdots
    & {\left((\sigma I - A)^{-\H}\right)}^{r} C^{\T}
  \end{bmatrix}
\end{align*}
```

This will achieve interpolation of the first $2 r$ moments at {math}`\sigma` of
the transfer function.

```{math}
\begin{align*}
  H(s)
  & = C {(s I - A)}^{-1} B + D \\
  & = C {((s - \sigma) I + \sigma I - A)}^{-1} B + D \\
  & = C
    {\left((\sigma I - A)
      \left((s - \sigma) (\sigma I - A)^{-1} + I\right)\right)}^{-1}
    B + D \\
  & = C
    {\left((s - \sigma) (\sigma I - A)^{-1} + I\right)}^{-1}
    {(\sigma I - A)}^{-1}
    B + D \\
  & = C
    {\left(I - (s - \sigma) (A - \sigma I)^{-1}\right)}^{-1}
    {(\sigma I - A)}^{-1}
    B + D \\
  & = C
    \sum_{k = 0}^{\infty} {\left((s - \sigma) (A - \sigma I)^{-1} \right)}^{k}
    {(\sigma I - A)}^{-1}
    B + D \\
  & = D - \sum_{k = 0}^{\infty} C (A - \sigma I)^{-(k + 1)} B (s - \sigma)^{k}.
\end{align*}
```

The moments at {math}`\sigma` are

```{math}
M_0(\sigma) = C {(\sigma I - A)}^{-1} B + D, \quad
M_k(\sigma) = -C {(A - \sigma I)}^{-(k + 1)} B,  \text{ for } k \ge 1.
```

To preserve realness in the ROMs, we interpolate at a conjugate pair of points.

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
for (i, rom), marker in zip(enumerate(roms), markers):
    r = 2 * (i + 1)
    poles = rom.poles()
    ax.plot(poles.real, poles.imag, marker, label=f'ROM $r = {r}$')
_ = ax.legend()
```

We observe that some of the ROMs are unstable.
In particular, interpolation does not necessarily preserve stability,
just as Petrov-Galerkin projection in general.

Download the code:
{download}`tutorial_interpolation.md`,
{nb-download}`tutorial_interpolation.ipynb`.
