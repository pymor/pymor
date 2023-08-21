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

# Tutorial: Model order reduction for port-Hamiltonian systems

In the [first section](#port-hamiltonian-lti-systems), we introduce the class of
port-Hamiltonian systems and their relationship to two other system-theoretic properties called
*passivity* and *positive realness*. After introducing a toy example in
the [second section](#a-toy-problem-mass-spring-damper-chain), we
have a look into structure-preserving model reduction for port-Hamiltonian systems
in the [third section](#structure-preserving-model-order-reduction).

## Port-Hamiltonian LTI systems

In this section, we give a formal definition of port-Hamiltonian systems.
Port-Hamiltonian systems have several favorable properties for modeling, control and
simulation, for example composability and stability. Furthermore, they adhere to a
certain power balance equation. Port-Hamiltonian systems are especially suited for
network-based modeling and problems involving multi-physics phenomena. We refer to {cite}`MU23`
for a general introduction to port-Hamiltonian systems and their applications.

We say a LTI system is *port-Hamiltonian* if it can be expressed as

```{math}
E \dot{x}(t) & = (J - R) Q x(t) + (G-P) u(t), \\
y(t) & = (G+P)^T Q x(t) + (S-N) u(t),
```

where {math}`H := Q^T E`, and if

```{math}
\Gamma =
\begin{bmatrix}
    J & G \\
    -G^T & N
\end{bmatrix},
\text{ and }
W =
\begin{bmatrix}
    R & P \\
    P^T & S
\end{bmatrix}
```

satisfy
{math}`H = H^T \succ 0`,
{math}`\Gamma^T = -\Gamma`, and
{math}`W = W^T \succcurlyeq 0`.

The quadratic (energy) function {math}`\mathcal{H}(x) := \tfrac{1}{2} x^T H x`,
typically called Hamiltonian, corresponds to the energy stored in the system. In
applications, {math}`E` and/or {math}`Q` often are identity matrices.

It is known that if the LTI system is minimal and stable, the following are equivalent:

- The system is passive.
- The system is port-Hamiltonian.
- The system is positive real.

See for example {cite}`BU22` for more details.

In pyMOR, there exists a {{ PHLTIModel }} class. As of now, pyMOR only supports
port-Hamiltonian systems with nonsingular E. {{ PHLTIModel }} inherits from
{{ LTIModel }}, so {{ PHLTIModel }} can be used with all reductors which expect
a {{ LTIModel }}. For model reduction, it is often desirable to preserve the
port-Hamiltonian structure, i.e., to compute a ROM which is also port-Hamiltonian.

A passive {{ LTIModel }} can be converted into a {{ PHLTIModel }} using
the {meth}`~pymor.models.iosys.PHLTIModel.from_passive_LTIModel` method if desired.
Consequentely, one option to preserve port-Hamiltonian structure is to use a reductor
which preserves passivity (but returns a ROM of type {{ LTIModel }}), and convert the
ROM into a {{ PHLTIModel }} in a post-processing step.

## A toy problem: Mass-spring-damper chain

As a toy problem, we use a mass-spring-damper chain, which can be formulated
as a port-Hamiltonian system (see {cite}`GPBV12`):

```{image} msd_example.svg
:alt: MSD example
:width: 60%
```

Here, the spring constants are denoted by {math}`k_i` and the damping constants by
{math}`c_i`, {math}`i=1,\dots,n/2`.
The inputs {math}`u_1` and {math}`u_2` are the external forces on the first two
masses {math}`m_1` and {math}`m_2`. The system outputs {math}`y_1` and {math}`y_2`
correspond to the velocities of the first two masses {math}`m_1` and {math}`m_2`.
The toy problem is included in pyMOR in the {mod}`pymor.models.examples` module as
{func}`~pymor.models.examples.msd_example`.

## Structure-preserving model order reduction

pyMOR provides three reductors which can be used for model order reduction
while preserving the port-Hamiltonian structure:

- pH-IRKA ({class}`~pymor.reductors.ph.ph_irka.PHIRKAReductor`),
- PRBT ({class}`~pymor.reductors.bt.PRBTReductor`),
- passivity preserving model reduction via spectral factorization
  ({class}`~pymor.reductors.spectral_factor.SpectralFactorReductor`).

In this section, we apply all three reductors on our toy example and compare
their performance. All three reductors are described in {cite}`BU22` in more detail.

Note: Currently, the {class}`~pymor.reductors.bt.PRBTReductor` and
{class}`~pymor.reductors.spectral_factor.SpectralFactorReductor` reductors require
{math}`D` to be nonsingular. The MSD example has a zero {math}`D` matrix. Therefore,
we have to add a small regularization feedthrough term. This is a limitation of the
current implementation, since in the background the KYP-LMI is converted
into a Riccati equation, which is only possible for nonsingular {math}`D`.
For {func}`~pymor.models.iosys.PHLTIModel.from_passive_LTIModel`, {math}`D` must be
nonsingular for the same reasons.

```{code-cell}
import numpy as np
from pymor.models.iosys import PHLTIModel
from pymor.models.examples import msd_example

J, R, G, P, S, N, E, Q = msd_example(50, 2)

# tolerance for solving the Riccati equation instead of KYP-LMI
# by introducing a regularization feedthrough term D
# (required for PRBTReductor and SpectralFactorReductor reductors)
S += np.eye(S.shape[0]) * 1e-12

fom = PHLTIModel.from_matrices(J, R, G, S=S, Q=Q, solver_options={'ricc_pos_lrcf': 'scipy'})
```

The `ricc_pos_lrcf` solver option refers to the solver used for the underlying
Riccati equation relevant for {class}`~pymor.reductors.bt.PRBTReductor` and
{class}`~pymor.reductors.spectral_factor.SpectralFactorReductor`. Possible choices are
`scipy`, `slycot` or `pymess` (if installed).

### pH-IRKA

The pH-IRKA reductor {class}`~pymor.reductors.ph.ph_irka.PHIRKAReductor` directly returns
a ROM of type {{ PHLTIModel }}. pH-IRKA works similar to the standard IRKA reductor
{class}`~pymor.reductors.h2.IRKAReductor`, but with fewer degrees of freedom to preserve
the port-Hamiltonian structure.

```{code-cell}
from pymor.reductors.ph.ph_irka import PHIRKAReductor

reductor = PHIRKAReductor(fom)
rom1 = reductor.reduce(10)
print(f'rom1 is of type {type(rom1)}.')
```

### Positive-real balanced truncation (PRBT)

Positive-real balanced truncation (PRBT) works similar to the general balanced truncation
method described in {doc}`tutorial_bt`, but uses positive real controllability
and observability Gramians instead. PRBT preserves passivity, but returns a ROM
of type {{ LTIModel }}. Thus, we convert the ROM into a {{ PHLTIModel }} in a
post-processing step. PRBT can be used with any passive {{ LTIModel}} FOM.

```{code-cell}
from pymor.reductors.bt import PRBTReductor

reductor = PRBTReductor(fom)
rom2 = reductor.reduce(10)
rom2 = PHLTIModel.from_passive_LTIModel(rom2)
print(f'rom2 is of type {type(rom1)}.')
```

### Passivity preserving model reduction via spectral factorization

The {class}`~pymor.reductors.spectral_factor.SpectralFactorReductor` method
is a wrapper reductor for another generic reductor. The method extracts a
spectral factor from the FOM, which subsequentely is reduced by a reductor
specified by the user. A spectral factor is a standard {{ LTIModel }}. Here, we opt
for {class}`~pymor.reductors.h2.IRKAReductor` as the inner reductor. If the inner
reductor returns a stable ROM, passivity is preserved.
The spectral factor method and PRBT are related, since the computation of
the optimal spectral factor for model reduction depends on the computation of
the positive real observability Gramian. The spectral factor method can be used with
any passive {{ LTIModel}} FOM. Again, we convert the ROM of type {{ LTIModel}} into a
{{ PHLTIModel }} in a post-processing step.

```{code-cell}
from pymor.reductors.spectral_factor import SpectralFactorReductor
from pymor.reductors.h2 import IRKAReductor

reductor = SpectralFactorReductor(fom)
rom3 = reductor.reduce(
    lambda spectral_factor, mu : IRKAReductor(spectral_factor, mu).reduce(10)
)
rom3 = PHLTIModel.from_passive_LTIModel(rom3)
print(f'rom3 is of type {type(rom1)}.')
```

### Comparison

Let us compare the {math}`\mathcal{H}_2` errors of the three methods:

```{code-cell}
err1 = fom - rom1
err2 = fom - rom2
err3 = fom - rom3

print(f'pHIRKA - Relative H2 error: {err1.h2_norm() / fom.h2_norm():.3e}')
print(f'PRBT - Relative H2 error: {err2.h2_norm() / fom.h2_norm():.3e}')
print(f'spectral_factor - Relative H2 error: {err3.h2_norm() / fom.h2_norm():.3e}')
```

We can plot a magnitude plot of the three error systems:

```{code-cell}
import matplotlib.pyplot as plt
w = (1e-4, 1e3)
fig, ax = plt.subplots()
err1.transfer_function.mag_plot(w, ax=ax, label='pHIRKA')
err2.transfer_function.mag_plot(w, ax=ax, linestyle='--', label='PRBT')
err3.transfer_function.mag_plot(w, ax=ax, label='spectral_factor')
_ = ax.legend()
```

Download the code:
{download}`tutorial_ph.md`,
{nb-download}`tutorial_ph.ipynb`.
