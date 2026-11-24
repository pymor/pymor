---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:load: myst_code_init.py
:tags: [remove-cell]


```

# Tutorial: {math}`\mathcal{H}_2`-optimal model order reduction

In {doc}`tutorial_bt`, we saw that balanced truncation produces reduced-order
models with an a priori {math}`\mathcal{H}_\infty` error bound.
However, these reduced-order models are in general not optimal with respect to
any system norm.
In this tutorial, we discuss methods which aim to find reduced-order models
that are (locally) optimal in the {math}`\mathcal{H}_2` norm:

- the iterative rational Krylov algorithm (IRKA) {cite}`GAB08`
  for LTI systems given in state-space form and
- the transfer function IRKA (TF-IRKA) {cite}`BG12`,
  which only needs evaluations of the transfer function and its derivative.

Both methods are based on bitangential Hermite interpolation,
which we discussed in {doc}`tutorial_interpolation`.
For a comprehensive treatment of these and related methods,
see {cite}`ABG20`.

We start by importing some packages and
silencing pyMOR's info messages to keep the output short.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from pymor.core.logger import set_log_levels

set_log_levels({'pymor': 'WARN'})
plt.rcParams['axes.grid'] = True
```

## The {math}`\mathcal{H}_2`-optimal model order reduction problem

We consider an asymptotically stable LTI system

```{math}
\begin{align}
    E \dot{x}(t) & = A x(t) + B u(t), \\
    y(t) & = C x(t),
\end{align}
```

with transfer function {math}`H(s) = C {(s E - A)}^{-1} B`,
input {math}`u(t) \in \mathbb{R}^m`, and output {math}`y(t) \in \mathbb{R}^p`.
We omit the feedthrough term {math}`D u(t)`, since the
{math}`\mathcal{H}_2` norm is only finite for {math}`D = 0`
(for {math}`D \neq 0`, the reduced-order model simply keeps the same {math}`D`).
As discussed in {doc}`tutorial_lti_systems`,
the {math}`\mathcal{H}_2` norm

```{math}
\lVert H \rVert_{\mathcal{H}_2}
=
\left(
  \frac{1}{2 \pi}
  \int_{-\infty}^{\infty}
  \lVert H(\boldsymbol{\imath} \omega) \rVert_{\operatorname{F}}^2
  \operatorname{d}\!\omega
\right)^{\frac{1}{2}}
```

bounds the {math}`\mathcal{L}_\infty` norm of the output in terms of the
{math}`\mathcal{L}_2` norm of the input.
Applied to the error system,
this gives for a reduced-order model with transfer function {math}`\hat{H}`
and output {math}`\hat{y}` that

```{math}
\lVert y - \hat{y} \rVert_{\mathcal{L}_\infty}
\leqslant
\lVert H - \hat{H} \rVert_{\mathcal{H}_2}
\lVert u \rVert_{\mathcal{L}_2}.
```

Therefore, a reduced-order model with a small {math}`\mathcal{H}_2` error
produces outputs that are uniformly close to the outputs of the full-order model
for all inputs of bounded energy.
This motivates the {math}`\mathcal{H}_2`-optimal model order reduction problem:
find an asymptotically stable reduced-order model of order {math}`r`
that minimizes {math}`\lVert H - \hat{H} \rVert_{\mathcal{H}_2}`.
This optimization problem is nonconvex and generally has multiple local
minima.
Therefore, we are interested in methods that find reduced-order models
satisfying the first-order necessary optimality conditions.

## Interpolatory optimality conditions

Assume that the reduced-order model has simple poles
{math}`\hat{\lambda}_1, \ldots, \hat{\lambda}_r`.
Then its transfer function has the pole-residue form

```{math}
\hat{H}(s) = \sum_{i = 1}^r \frac{\hat{c}_i \hat{b}_i^{\operatorname{T}}}{s - \hat{\lambda}_i},
```

where {math}`\hat{c}_i \in \mathbb{C}^p` and {math}`\hat{b}_i \in \mathbb{C}^m`
are the left and right residue directions.
If {math}`\hat{H}` is a locally {math}`\mathcal{H}_2`-optimal
reduced-order model,
then (see {cite}`GAB08`)

```{math}
\begin{align}
    H\big(-\hat{\lambda}_i\big) \hat{b}_i
    & = \hat{H}\big(-\hat{\lambda}_i\big) \hat{b}_i, \\
    \hat{c}_i^{\operatorname{T}} H\big(-\hat{\lambda}_i\big)
    & = \hat{c}_i^{\operatorname{T}} \hat{H}\big(-\hat{\lambda}_i\big), \\
    \hat{c}_i^{\operatorname{T}} H'\big(-\hat{\lambda}_i\big) \hat{b}_i
    & = \hat{c}_i^{\operatorname{T}} \hat{H}'\big(-\hat{\lambda}_i\big) \hat{b}_i,
\end{align}
```

for {math}`i = 1, \ldots, r`.
In other words,
{math}`\hat{H}` has to be a bitangential Hermite interpolant of {math}`H`
at the mirror images of its own poles,
with tangential directions given by its own residue directions.
For single-input single-output systems,
these conditions simplify to
{math}`H(-\hat{\lambda}_i) = \hat{H}(-\hat{\lambda}_i)` and
{math}`H'(-\hat{\lambda}_i) = \hat{H}'(-\hat{\lambda}_i)`.

In {doc}`tutorial_interpolation`, we saw how to construct a bitangential
Hermite interpolant for given interpolation points and tangential directions.
The difficulty here is that the optimal interpolation points and tangential
directions depend on the reduced-order model we are looking for.

## Iterative rational Krylov algorithm (IRKA)

IRKA resolves this using a fixed-point iteration.
Starting from initial interpolation points
{math}`\sigma_1, \ldots, \sigma_r`
and tangential directions
{math}`b_1, \ldots, b_r` and {math}`c_1, \ldots, c_r`
(closed under complex conjugation),
it repeats the following steps:

1. Compute the basis matrices

   ```{math}
   \begin{align}
       V & =
       \begin{bmatrix}
           {(\sigma_1 E - A)}^{-1} B b_1 & \cdots & {(\sigma_r E - A)}^{-1} B b_r
       \end{bmatrix}, \\
       W & =
       \begin{bmatrix}
           {(\sigma_1 E - A)}^{-\operatorname{T}} C^{\operatorname{T}} c_1
           & \cdots
           & {(\sigma_r E - A)}^{-\operatorname{T}} C^{\operatorname{T}} c_r
       \end{bmatrix}.
   \end{align}
   ```

2. Compute the reduced-order model using Petrov-Galerkin projection, i.e.,
   {math}`\hat{E} = W^{\operatorname{T}} E V`,
   {math}`\hat{A} = W^{\operatorname{T}} A V`,
   {math}`\hat{B} = W^{\operatorname{T}} B`,
   {math}`\hat{C} = C V`.
   Its transfer function is a bitangential Hermite interpolant of {math}`H`
   at the points {math}`\sigma_i` with directions {math}`b_i` and {math}`c_i`
   (see {doc}`tutorial_interpolation`).
3. Compute the poles {math}`\hat{\lambda}_i` and
   residue directions {math}`\hat{b}_i`, {math}`\hat{c}_i` of the
   reduced-order model.
4. Set {math}`\sigma_i \gets -\hat{\lambda}_i`,
   {math}`b_i \gets \hat{b}_i`, and
   {math}`c_i \gets \hat{c}_i`.

In practice, {math}`V` and {math}`W` are replaced by real bases of the same
subspaces.
If the iteration converges,
the resulting reduced-order model satisfies the optimality conditions above.
Convergence is not guaranteed in general,
but IRKA converges in many practical applications.
Compared to balanced truncation,
each IRKA iteration only requires solving {math}`2 r` linear systems with
shifted matrices {math}`\sigma_i E - A` and no Lyapunov equations,
which makes it attractive for large-scale systems.

## IRKA in pyMOR

We demonstrate IRKA on
[Penzl's FOM example](https://modelreduction.org/index.php/Penzl%27s_FOM),
which we also used in {doc}`tutorial_interpolation`.

```{code-cell} ipython3
from pymor.models.examples import penzl_example

fom = penzl_example()
print(fom)
```

Its magnitude plot shows three peaks.

```{code-cell} ipython3
w = (1e-1, 1e4)
_ = fom.transfer_function.mag_plot(w)
```

To run IRKA, we create an {class}`~pymor.reductors.h2.IRKAReductor` and call
its {meth}`~pymor.reductors.h2.IRKAReductor.reduce` method.
Its first argument can be the reduced order,
initial interpolation points,
a dictionary of initial interpolation points and tangential directions,
or an initial reduced-order model.
Passing the reduced order uses default initial interpolation data.
By setting `compute_errors=True`, the relative {math}`\mathcal{H}_2` errors of
all intermediate reduced-order models are computed.
This is useful for illustration,
but should otherwise be avoided since it is expensive for large systems.

```{code-cell} ipython3
from pymor.reductors.h2 import IRKAReductor

irka = IRKAReductor(fom)
rom_irka = irka.reduce(10, compute_errors=True)
```

After the reduction,
the reductor stores the convergence criterion values in `irka.conv_crit`
(by default, the relative change in interpolation points),
the relative errors in `irka.errors`,
and the interpolation points and tangential directions of each iteration in
`irka.sigma_list`, `irka.b_list`, and `irka.c_list`.

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.semilogy(range(1, len(irka.conv_crit) + 1), irka.conv_crit, '.-', label='Convergence criterion')
ax.semilogy(range(1, len(irka.errors) + 1), irka.errors, '.-', label=r'Relative $\mathcal{H}_2$ error')
ax.set_xlabel('Iteration')
_ = ax.legend()
```

The convergence criterion decreases monotonically after a few iterations,
until it falls below the tolerance (`tol=1e-4` by default).
The missing points in the error curve correspond to intermediate
reduced-order models that are unstable,
for which pyMOR reports the error as infinite.
Neither stability of intermediate reduced-order models
nor monotonic decrease of the error is guaranteed.

We can also compare the initial and final interpolation points with the mirror
images of the full-order poles.
Computing all full-order poles requires a dense eigenvalue decomposition
(pyMOR warns that it converts {math}`A` to a dense matrix),
which is feasible here since the full-order model is small.

```{code-cell} ipython3
sigma_init = irka.sigma_list[0]
sigma_final = irka.sigma_list[-1]
fom_poles_mirrored = -fom.poles()

fig, ax = plt.subplots()
ax.semilogx(fom_poles_mirrored.real, fom_poles_mirrored.imag, '.', markersize=2, label='Mirrored FOM poles')
ax.semilogx(sigma_init.real, sigma_init.imag, 'x', label='Initial points')
ax.semilogx(sigma_final.real, sigma_final.imag, 'o', fillstyle='none', label='Final points')
ax.set_xlabel('Real part')
ax.set_ylabel('Imaginary part')
_ = ax.legend()
```

IRKA started from real interpolation points and moved six of them to the
mirror images of the three pairs of complex poles responsible for the peaks in
the magnitude plot.
The remaining four are spread over the real axis.

Next, we check that the optimality conditions are satisfied.
Since Penzl's example has a single input and a single output,
we only need to compare the transfer functions and their derivatives at the
mirror images of the reduced-order poles.

```{code-cell} ipython3
print('Interpolation point      Rel. error in H   Rel. error in dH')
for s in -rom_irka.poles():
    H = fom.transfer_function.eval_tf(s)
    Hr = rom_irka.transfer_function.eval_tf(s)
    dH = fom.transfer_function.eval_dtf(s)
    dHr = rom_irka.transfer_function.eval_dtf(s)
    rel_err = np.linalg.norm(H - Hr) / np.linalg.norm(H)
    rel_err_d = np.linalg.norm(dH - dHr) / np.linalg.norm(dH)
    print(f'{s:22.3f}  {rel_err:17.2e}  {rel_err_d:17.2e}')
```

The interpolation conditions are satisfied up to small errors,
which come from stopping the iteration at a finite tolerance:
the reduced-order model interpolates exactly at the interpolation points
of the last iteration, which are close to, but not exactly equal to,
the mirror images of its poles.

Let us compare the magnitude plots of the full-order and reduced-order models

```{code-cell} ipython3
fig, ax = plt.subplots()
fom.transfer_function.mag_plot(w, ax=ax, label='FOM')
rom_irka.transfer_function.mag_plot(w, ax=ax, linestyle='--', label='IRKA ROM')
_ = ax.legend()
```

and draw the magnitude plot of the error system.

```{code-cell} ipython3
err_irka = fom - rom_irka
_ = err_irka.transfer_function.mag_plot(w)
```

### Comparison with balanced truncation

Next, we compare IRKA with balanced truncation for different reduced orders.
Since IRKA only finds local minima,
the result can depend on the initial interpolation data.
Therefore, we run IRKA with the default initialization as well as with the
balanced truncation reduced-order model as the initial reduced-order model.

```{code-cell} ipython3
from pymor.reductors.bt import BTReductor

bt = BTReductor(fom)
fom_h2_norm = fom.h2_norm()
orders = range(2, 17, 2)
errors_bt = []
errors_irka = []
errors_irka_bt = []
for r in orders:
    rom_bt = bt.reduce(r)
    rom_irka_r = IRKAReductor(fom).reduce(r)
    rom_irka_bt = IRKAReductor(fom).reduce(rom_bt)
    errors_bt.append((fom - rom_bt).h2_norm() / fom_h2_norm)
    errors_irka.append((fom - rom_irka_r).h2_norm() / fom_h2_norm)
    errors_irka_bt.append((fom - rom_irka_bt).h2_norm() / fom_h2_norm)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.semilogy(orders, errors_bt, '.-', label='BT')
ax.semilogy(orders, errors_irka, 'x--', label='IRKA (default initialization)')
ax.semilogy(orders, errors_irka_bt, '+:', label='IRKA (initialized with BT)')
ax.set_xlabel('Reduced order')
ax.set_title(r'Relative $\mathcal{H}_2$ errors')
_ = ax.legend()
```

For most orders,
IRKA finds reduced-order models with smaller {math}`\mathcal{H}_2` errors than
balanced truncation.
An exception is the order {math}`r = 6`,
where IRKA with the default initialization converges to a local minimum
with a larger error than balanced truncation.
Initializing IRKA with the balanced truncation reduced-order model avoids this
here.

```{code-cell} ipython3
print(f'r = 6, BT:                  {errors_bt[2]:.3e}')
print(f'r = 6, IRKA (default init): {errors_irka[2]:.3e}')
print(f'r = 6, IRKA (BT init):      {errors_irka_bt[2]:.3e}')
```

The {meth}`~pymor.reductors.h2.IRKAReductor.reduce` method has further
parameters to influence the iteration, e.g.,

- `tol` and `maxit` for the tolerance and maximum number of iterations,
- `conv_crit='h2'` for using the relative {math}`\mathcal{H}_2` distance
  between consecutive reduced-order models as the convergence criterion,
- `num_prev` for comparing to more than one previous iteration,
  which can help when IRKA cycles between multiple points,
- `force_sigma_in_rhp=True` for reflecting only the reduced-order poles in the
  left half-plane, to keep the interpolation points in the right half-plane, and
- `projection` for choosing how the basis matrices are computed
  (`'orth'`, `'biorth'`, or, for single-input single-output systems, `'arnoldi'`).

## Transfer function IRKA (TF-IRKA)

IRKA needs a state-space realization of the full-order model
to solve linear systems with {math}`\sigma_i E - A`.
However, the optimality conditions only involve evaluations of {math}`H` and
{math}`H'`.
This is the idea behind TF-IRKA {cite}`BG12`,
which replaces the projection step in IRKA by the Loewner framework.
Given interpolation points {math}`\sigma_i`
and tangential directions {math}`b_i`, {math}`c_i`,
the reduced-order model with

```{math}
\begin{align}
    \hat{E}_{ij} & =
    \begin{cases}
        -\dfrac{c_i^{\operatorname{T}} (H(\sigma_i) - H(\sigma_j)) b_j}{\sigma_i - \sigma_j},
        & i \neq j, \\
        -c_i^{\operatorname{T}} H'(\sigma_i) b_i,
        & i = j,
    \end{cases} \\
    \hat{A}_{ij} & =
    \begin{cases}
        -\dfrac{c_i^{\operatorname{T}} (\sigma_i H(\sigma_i) - \sigma_j H(\sigma_j)) b_j}{\sigma_i - \sigma_j},
        & i \neq j, \\
        -c_i^{\operatorname{T}} (H(\sigma_i) + \sigma_i H'(\sigma_i)) b_i,
        & i = j,
    \end{cases} \\
    \hat{B} & =
    \begin{bmatrix}
        c_1^{\operatorname{T}} H(\sigma_1) \\
        \vdots \\
        c_r^{\operatorname{T}} H(\sigma_r)
    \end{bmatrix}, \\
    \hat{C} & =
    \begin{bmatrix}
        H(\sigma_1) b_1 & \cdots & H(\sigma_r) b_r
    \end{bmatrix}
\end{align}
```

satisfies the same bitangential Hermite interpolation conditions as the
reduced-order model computed by projection in IRKA
(assuming that the matrices {math}`\sigma_i \hat{E} - \hat{A}` are invertible).
In pyMOR, this construction is implemented in
{class}`~pymor.reductors.interpolation.TFBHIReductor`.
TF-IRKA uses it in the same fixed-point iteration as IRKA.
Therefore, TF-IRKA can be applied to any system for which
{math}`H(s)` and {math}`H'(s)` can be evaluated,
including systems without a (known) finite-dimensional state-space realization.

### TF-IRKA for an LTI system

To check that TF-IRKA gives the same result as IRKA,
we apply {class}`~pymor.reductors.h2.TFIRKAReductor` to Penzl's example.
It accepts a {class}`~pymor.models.transfer_function.TransferFunction`
or any model with a `transfer_function` attribute,
and only uses the transfer function and its derivative.

```{code-cell} ipython3
from pymor.reductors.h2 import TFIRKAReductor

tf_irka = TFIRKAReductor(fom.transfer_function)
rom_tf_irka = tf_irka.reduce(10)
```

Starting from the same initial interpolation data,
TF-IRKA needs the same number of iterations as IRKA
and finds a reduced-order model with the same {math}`\mathcal{H}_2` error.

```{code-cell} ipython3
err_tf_irka = fom - rom_tf_irka
print(f'IRKA:    {len(irka.conv_crit)} iterations, '
      f'relative H2 error {err_irka.h2_norm() / fom_h2_norm:.4e}')
print(f'TF-IRKA: {len(tf_irka.conv_crit)} iterations, '
      f'relative H2 error {err_tf_irka.h2_norm() / fom_h2_norm:.4e}')
```

The two methods are equivalent in exact arithmetic.
In floating-point arithmetic,
their intermediate iterates can differ significantly,
since the interpolation problems in the first iterations are ill-conditioned.
Nevertheless, both converge to the same reduced-order model here.

### TF-IRKA for a time-delay system

As an example without a finite-dimensional state-space realization,
we consider the time-delay system

```{math}
\begin{align}
    \dot{x}(t) & = a x(t) + u(t - \tau), \\
    y(t) & = x(t),
\end{align}
```

with transfer function

```{math}
H(s) = \frac{e^{-\tau s}}{s - a}.
```

Its derivative is

```{math}
H'(s) = -\frac{(\tau (s - a) + 1) e^{-\tau s}}{{(s - a)}^2}.
```

We can create a {class}`~pymor.models.transfer_function.TransferFunction`
by providing the number of inputs and outputs and
functions that evaluate {math}`H` and {math}`H'`.
The functions have to return 2D {{ NumPy_arrays }}.

```{code-cell} ipython3
from pymor.models.transfer_function import TransferFunction

tau = 1
a = -0.1


def H(s):
    return np.array([[np.exp(-tau * s) / (s - a)]])


def dH(s):
    return np.array([[-(tau * (s - a) + 1) * np.exp(-tau * s) / (s - a)**2]])


tf = TransferFunction(1, 1, H, dH, name='TimeDelaySystem')
print(tf)
```

This transfer function is also available as
{func}`~pymor.models.examples.transfer_function_delay_example`.
Let us draw its Bode plot.

```{code-cell} ipython3
w_delay = (1e-2, 1e2)
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(6, 8), sharex=True, constrained_layout=True)
_ = tf.bode_plot(w_delay, ax=ax)
```

The phase decreases without bound due to the time delay.
Next, we run TF-IRKA.
For this example, TF-IRKA needs more iterations to converge,
so we increase the maximum number of iterations.

```{code-cell} ipython3
tf_irka_delay = TFIRKAReductor(tf)
rom_delay = tf_irka_delay.reduce(8, maxit=1000)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.semilogy(range(1, len(tf_irka_delay.conv_crit) + 1), tf_irka_delay.conv_crit)
ax.set_xlabel('Iteration')
_ = ax.set_title('Convergence criterion')
```

After an initial phase,
the convergence criterion decreases linearly, but slowly.
We compare the Bode plots of the full-order and reduced-order models

```{code-cell} ipython3
fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(6, 8), sharex=True, constrained_layout=True)
tf.bode_plot(w_delay, ax=ax, label='FOM')
rom_delay.transfer_function.bode_plot(w_delay, ax=ax, linestyle='--', label='TF-IRKA ROM')
_ = ax[0, 0].legend()
_ = ax[1, 0].legend()
```

and draw the magnitude plot of the error.

```{code-cell} ipython3
err_delay = tf - rom_delay
_ = err_delay.mag_plot(w_delay)
```

The reduced-order model approximates the full-order model well for low
frequencies.
For high frequencies,
the phase of a rational transfer function of order {math}`r` is bounded,
so the reduced-order model cannot follow the phase of the time-delay system.
However, since the magnitude decays,
the high frequencies contribute less to the {math}`\mathcal{H}_2` error.

Since the time-delay system does not have a state-space realization,
the {meth}`~pymor.models.transfer_function.TransferFunction.h2_norm` method of
{class}`~pymor.models.transfer_function.TransferFunction` uses numerical
quadrature (via `scipy.integrate.quad`)
instead of solving Lyapunov equations.
This works well for {math}`H` itself,
but the integrand for the error {math}`H - \hat{H}` oscillates because of the
time delay, which makes the quadrature inaccurate.
Instead, we use the expansion

```{math}
\lVert H - \hat{H} \rVert_{\mathcal{H}_2}^2
=
\lVert H \rVert_{\mathcal{H}_2}^2
- 2 \operatorname{Re} \langle H, \hat{H} \rangle_{\mathcal{H}_2}
+ \lVert \hat{H} \rVert_{\mathcal{H}_2}^2.
```

The inner product can be computed using
{meth}`~pymor.models.transfer_function.TransferFunction.h2_inner`,
which only needs to evaluate {math}`H` at the mirror images of the poles of
{math}`\hat{H}` (see, e.g., {cite}`ABG10`, Lemma 1).
Let us compute the relative {math}`\mathcal{H}_2` errors for different
reduced orders.

```{code-cell} ipython3
tf_h2_norm = tf.h2_norm()
orders_delay = [2, 4, 6, 8]
errors_delay = []
for r in orders_delay:
    rom_r = TFIRKAReductor(tf).reduce(r, maxit=1000)
    err_h2_norm = np.sqrt(tf_h2_norm**2 - 2 * tf.h2_inner(rom_r).real + rom_r.h2_norm()**2)
    errors_delay.append(err_h2_norm / tf_h2_norm)

fig, ax = plt.subplots()
ax.semilogy(orders_delay, errors_delay, '.-')
ax.set_xlabel('Reduced order')
_ = ax.set_title(r'Relative $\mathcal{H}_2$ errors')
```

The error decreases much slower than for Penzl's example,
since the delay term {math}`e^{-\tau s}` is difficult to approximate by
rational functions on the imaginary axis.

## Further IRKA variants

pyMOR implements several other methods related to IRKA:

- {class}`~pymor.reductors.h2.TSIAReductor`, the two-sided iteration
  algorithm {cite}`XZ11,BKS11`, which is equivalent to IRKA in exact
  arithmetic, but uses Sylvester equations instead of the eigenvalue
  decomposition of the reduced-order model,
- {class}`~pymor.reductors.h2.OneSidedIRKAReductor`,
  which uses a Galerkin projection instead of a Petrov-Galerkin projection,
- {class}`~pymor.reductors.h2.GapIRKAReductor` for unstable systems
  (see {doc}`tutorial_unstable_lti_systems`), and
- {class}`~pymor.reductors.ph.ph_irka.PHIRKAReductor` for port-Hamiltonian
  systems (see {doc}`tutorial_ph`).

Download the code:
{download}`tutorial_h2.md`,
{nb-download}`tutorial_h2.ipynb`.
