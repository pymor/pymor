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

## Port-Hamiltonian LTI systems

```{math}
E \dot{x}(t) & = (J - R) Q x(t) + (G-P) u(t), \\
y(t) & = (G+P)^T Q x(t) + (S-N) u(t),
```

where {math}`H := Q^T E`,

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

A dynamical system of this form, together with a given quadratic (energy)
function {math}`\mathcal{H}(x) := \tfrac{1}{2} x^T H x`, typically
called Hamiltonian, is called a port-Hamiltonian system.

In pyMOR, there exists {{ PHLTIModel }}. As of now, pyMOR only supports
port-Hamiltonian systems with nonsingular E. {{ PHLTIModel }} inherits from
{{ LTIModel }}, so {{ PHLTIModel }} can be used with all reductors which expect
a {{ LTIModel }}. However, only specific reductors preserve the
port-Hamiltonian structure, i.e., return a ROM of type {{ PHLTIModel }} or
allow the conversion of the ROM to {{ PHLTIModel }}. In applications, it is
often desirable to preserve the port-Hamiltonian structure.

It is known that if the LTI system is minimal and stable, the following are equivalent:
- The system is passive.
- The system is port-Hamiltonian.
- The system is positive real.

See for example {cite}`BU22` for more details.

## A toy problem: Mass-spring-damper

{cite}`GPBV12`

## Conversion of passive LTIModel to PHLTIModel

Note: Currently `from_passive_LTIModel` requires {math}`D` to be nonsingular.
The MSD problem has a zero {math}`D` matrix. Therefore, we have to add a small
regularization feedthrough term.

```{code-cell}
import numpy as np
from pymor.models.examples import msd_example
from pymor.models.iosys import LTIModel, PHLTIModel

A, B, C, D, E = msd_example(50, 2, as_lti=True)

# tolerance for solving the Riccati equation instead of KYP-LMI
# by introducing a regularization feedthrough term D
D = np.eye(D.shape[0]) * 1e-12

fom2 = LTIModel.from_matrices(A, B, C, D)
fom2_ph = PHLTIModel.from_passive_LTIModel(fom2)

print(f'Type of fom2: {type(fom2)}')
print(f'Type of fom2_ph: {type(fom2_ph)}')

w = (1e-4, 1e3)
err = fom2_ph - fom2
_ = err.transfer_function.mag_plot(w)
```

Theoretically the same (numercial error).

## Model order reduction

Passivity preserving.
The pHIRKA reductor directly returns a ROM of type {{ PHLTIModel }}.
If the reductor returns a passive ROM of type {{ LTIModel }}, it can be
converted to {{ PHLTIModel }} as described above.

```{code-cell}
J, R, G, P, S, N, E, Q = msd_example(50, 2)

# tolerance for solving the Riccati equation instead of KYP-LMI
# by introducing a regularization feedthrough term D
# (required for PRBT and spectral_factor)
S += np.eye(S.shape[0]) * 1e-12

fom = PHLTIModel.from_matrices(J, R, G, S=S, Q=Q, solver_options={'ricc_pos_lrcf': 'scipy'})
```

```{code-cell}
import matplotlib.pyplot as plt
_ = fom.transfer_function.mag_plot(w)
```

### pHIRKA

```{code-cell}
from pymor.reductors.ph.ph_irka import PHIRKAReductor

reductor = PHIRKAReductor(fom)
rom1 = reductor.reduce(10)
```

### Positive-real balanced truncation (PRBT)

```{code-cell}
from pymor.reductors.bt import PRBTReductor

reductor = PRBTReductor(fom)
rom2 = reductor.reduce(10)
```

### Passivity preserving model reduction via spectral factorization

```{code-cell}
from pymor.reductors.spectral_factor import SpectralFactorReductor
from pymor.reductors.h2 import IRKAReductor

reductor = SpectralFactorReductor(fom)
rom3 = reductor.reduce(
    lambda spectral_factor, mu : IRKAReductor(spectral_factor, mu).reduce(10)
)
```

### Comparison

```{code-cell}
err1 = fom - rom1
err2 = fom - rom2
err3 = fom - rom3

print(f'pHIRKA - Relative H2 error: {err1.h2_norm() / fom.h2_norm():.3e}')
print(f'PRBT - Relative H2 error: {err2.h2_norm() / fom.h2_norm():.3e}')
print(f'spectral_factor - Relative H2 error: {err3.h2_norm() / fom.h2_norm():.3e}')
```

```{code-cell}
fig, ax = plt.subplots()
err1.transfer_function.mag_plot(w, ax=ax, label='pHIRKA')
err2.transfer_function.mag_plot(w, ax=ax, linestyle='--', label='PRBT')
err3.transfer_function.mag_plot(w, ax=ax, label='spectral_factor')
_ = ax.legend()
```
