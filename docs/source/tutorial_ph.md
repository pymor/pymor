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
a {{ LTIModel }}.

## MSD problem


```{code-cell}
import numpy as np
from pymor.models.iosys import PHLTIModel

def msd(n=6, m=2, m_i=4, k_i=4, c_i=1, as_lti=False):
    """Mass-spring-damper model as (port-Hamiltonian) linear time-invariant system.

    Taken from :cite:`GPBV12`.

    Parameters
    ----------
    n
        The order of the model.
    m
        The number or inputs and outputs of the model.
    m_i
        The weight of the masses.
    k_i
        The stiffness of the springs.
    c_i
        The amount of damping.
    as_lti
        If `True`, the matrices of the standard linear time-invariant system are returned.
        Otherwise, the matrices of the port-Hamiltonian linear time-invariant system are returned.

    Returns
    -------
    A
        The LTI |NumPy array| A, if `as_lti` is `True`.
    B
        The LTI |NumPy array| B, if `as_lti` is `True`.
    C
        The LTI |NumPy array| C, if `as_lti` is `True`.
    D
        The LTI |NumPy array| D, if `as_lti` is `True`.
    J
        The pH |NumPy array| J, if `as_lti` is `False`.
    R
        The pH |NumPy array| R, if `as_lti` is `False`.
    G
        The pH |NumPy array| G, if `as_lti` is `False`.
    P
        The pH |NumPy array| P, if `as_lti` is `False`.
    S
        The pH |NumPy array| S, if `as_lti` is `False`.
    N
        The pH |NumPy array| N, if `as_lti` is `False`.
    E
        The LTI |NumPy array| E, if `as_lti` is `True`, or
        the pH |NumPy array| E, if `as_lti` is `False`.
    """
    assert n % 2 == 0
    n //= 2

    A = np.array(
        [[0, 1 / m_i, 0, 0, 0, 0], [-k_i, -c_i / m_i, k_i, 0, 0, 0],
         [0, 0, 0, 1 / m_i, 0, 0], [k_i, 0, -2 * k_i, -c_i / m_i, k_i, 0],
         [0, 0, 0, 0, 0, 1 / m_i], [0, 0, k_i, 0, -2 * k_i, -c_i / m_i]])

    if m == 2:
        B = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0], [0, 0, 0, 1 / m_i, 0, 0]])
    elif m == 1:
        B = np.array([[0, 1, 0, 0, 0, 0]]).T
        C = np.array([[0, 1 / m_i, 0, 0, 0, 0]])
    else:
        assert False

    J_i = np.array([[0, 1], [-1, 0]])
    J = np.kron(np.eye(3), J_i)
    R_i = np.array([[0, 0], [0, c_i]])
    R = np.kron(np.eye(3), R_i)

    for i in range(4, n + 1):
        B = np.vstack((B, np.zeros((2, m))))
        C = np.hstack((C, np.zeros((m, 2))))

        J = np.block([
            [J, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), J_i]
        ])

        R = np.block([
            [R, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, (i - 1) * 2)), R_i]
        ])

        A = np.block([
            [A, np.zeros(((i - 1) * 2, 2))],
            [np.zeros((2, i * 2))]
        ])

        A[2 * i - 2, 2 * i - 2] = 0
        A[2 * i - 1, 2 * i - 1] = -c_i / m_i
        A[2 * i - 3, 2 * i - 2] = k_i
        A[2 * i - 2, 2 * i - 1] = 1 / m_i
        A[2 * i - 2, 2 * i - 3] = 0
        A[2 * i - 1, 2 * i - 2] = -2 * k_i
        A[2 * i - 1, 2 * i - 4] = k_i

    Q = np.linalg.solve(J - R, A)
    G = B
    P = np.zeros(G.shape)
    D = np.zeros((m, m))
    E = np.eye(2 * n)
    S = (D + D.T) / 2
    N = -(D - D.T) / 2

    if as_lti:
        return A, B, C, D, E

    return J, R, G, P, S, N, E, Q

J, R, G, P, S, N, E, Q = msd(50, 2)

# tolerance for solving the Riccati equation instead of KYP-LMI
# by introducing a regularization feedthrough term D
eps = 1e-12
S += np.eye(S.shape[0]) * eps

fom = PHLTIModel.from_matrices(J, R, G, S=S, Q=Q, solver_options={'ricc_pos_lrcf': 'slycot'})
```

```{code-cell}
import matplotlib.pyplot as plt
w = (1e-4, 1e3)
_ = fom.transfer_function.mag_plot(w)
```

## pHIRKA

```{code-cell}
from pymor.reductors.ph.ph_irka import PHIRKAReductor

reductor = PHIRKAReductor(fom)
rom = reductor.reduce(10)
```

```{code-cell}
err = fom - rom
_ = err.transfer_function.mag_plot(w)
```

```{code-cell}
print(f'Relative H2 error: {err.h2_norm() / fom.h2_norm():.3e}')
```

## Positive-real balanced truncation (PRBT)

```{code-cell}
from pymor.reductors.bt import PRBTReductor

reductor = PRBTReductor(fom)
rom = reductor.reduce(10)
```

```{code-cell}
err = fom - rom
_ = err.transfer_function.mag_plot(w)
```

```{code-cell}
print(f'Relative H2 error: {err.h2_norm() / fom.h2_norm():.3e}')
```

## Passivity preserving model reduction via spectral factorization

```{code-cell}
from pymor.reductors.spectral_factor import SpectralFactorReductor
from pymor.reductors.h2 import IRKAReductor

reductor = SpectralFactorReductor(fom)
rom = reductor.reduce(
    lambda spectral_factor, mu : IRKAReductor(spectral_factor, mu).reduce(10)
)
```

```{code-cell}
err = fom - rom
_ = err.transfer_function.mag_plot(w)
```

```{code-cell}
print(f'Relative H2 error: {err.h2_norm() / fom.h2_norm():.3e}')
```
