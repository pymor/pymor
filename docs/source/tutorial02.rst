Tutorial 2: Reducing a heat equation using balanced truncation
==============================================================

:jupyter-download:notebook:`tutorial02`
:jupyter-download:script:`tutorial02`


Heat equation
-------------

We consider the following one-dimensional heat equation over :math:`(0, 1)` with two inputs
:math:`u_1, u_2` and three outputs :math:`y_1, y_2, y_2`:

.. math::

    \begin{align}
        \partial_t T(\xi, t) & = \partial_{\xi \xi} T(\xi, t) + u_1(t),
        & 0 < \xi < 1,\ t > 0, \\
        -\partial_\xi T(0, t) & = -T(0, t) + u_2(t),
        & t > 0, \\
        \partial_\xi T(1, t) & = -T(1, t),
        & t > 0, \\
        y_1(t) & = T(0, t),
        & t > 0, \\
        y_2(t) & = T(0.5, t),
        & t > 0, \\
        y_3(t) & = T(1, t),
        & t > 0.
    \end{align}

In the following, we will create a discretized |Model| and reduce it using the
balanced truncation method to approximate the mapping from inputs
:math:`u = (u_1, u_2)` to outputs :math:`y = (y_1, y_2, y_3)`.

Discretized model
-----------------

We need to construct a linear time-invariant (LTI) system

.. math::

    \begin{align}
        E \dot{x}(t) & = A x(t) + B u(t), \\
        y(t) & = C x(t) + D u(t).
    \end{align}

In pyMOR, these models are captured by |LTIModels| from the
:mod:`pymor.models.iosys` module.

There are many ways of building an |LTIModel|.
Here, we will use its :meth:`~pymor.models.iosys.LTIModel.from_matrices` method,
which instantiates an |LTIModel| from NumPy or SciPy matrices.

First, we do the necessary imports.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sps
    from pymor.basic import LTIModel, BTReductor

Next, we can assemble the matrices based on a centered finite difference
approximation:

.. jupyter-execute::

    k = 50
    n = 2 * k + 1

    A = sps.diags(
        [(n - 1) * [(n - 1)**2], n * [-2 * (n - 1)**2], (n - 1) * [(n - 1)**2]],
        [-1, 0, 1],
        format='lil',
    )
    A[0, 0] = A[-1, -1] = -2 * n * (n - 1)
    A[0, 1] = A[-1, -2] = 2 * (n - 1)**2
    A = A.tocsc()

    B = np.zeros((n, 2))
    B[:, 0] = 1
    B[0, 1] = 2 * (n - 1)

    C = np.zeros((3, n))
    C[0, 0] = C[1, k] = C[2, -1] = 1

Then, we can create an |LTIModel|:

.. jupyter-execute::

    fom = LTIModel.from_matrices(A, B, C)

We can get the internal representation of the |LTIModel| `fom`

.. jupyter-execute::

    fom

From this, we see that the matrices were wrapped in |NumpyMatrixOperators|,
while default values were chosen for :math:`D` and :math:`E` matrices
(respectively, zero and identity). The operators in an |LTIModel| can be
accessed directly, e.g., `fom.A`.

We can also see some basic information from `fom`'s string representation

.. jupyter-execute::

    print(fom)

To visualize the behavior of the `fom`, we can draw its magnitude plot

.. jupyter-execute::

    w = np.logspace(-2, 8, 50)
    fom.mag_plot(w)
    plt.grid()

Plotting the Hankel singular values shows us how well an LTI system can be
approximated by a reduced-order model

.. jupyter-execute::

    hsv = fom.hsv()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    ax.set_title('Hankel singular values')
    ax.grid()

As expected for a heat equation, the Hankel singular values decay rapidly.


Running balanced truncation
---------------------------

First, we need the reductor object

.. jupyter-execute::

    bt = BTReductor(fom)

Calling its :meth:`~pymor.reductors.bt.GenericBTReductor.reduce` method runs the
balanced truncation algorithm. This reductor additionally has an `error_bounds`
method which can compute the a priori :math:`\mathcal{H}_\infty` error bounds
based on the Hankel singular values:

.. jupyter-execute::

    error_bounds = bt.error_bounds()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(error_bounds) + 1), error_bounds, '.-')
    ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
    ax.set_xlabel('Reduced order')
    ax.set_title(r'Upper and lower $\mathcal{H}_\infty$ error bounds')
    ax.grid()

To get a reduced-order model of order 10, we call the `reduce` method with the
appropriate argument:

.. jupyter-execute::

    rom = bt.reduce(10)

Instead, or in addition, a tolerance for the :math:`\mathcal{H}_\infty` error
can be specified, as well as the projection algorithm (by default, the
balancing-free square root method is used).

We can compare the magnitude plots between the full-order and reduced-order
models

.. jupyter-execute::

    fig, ax = plt.subplots()
    fom.mag_plot(w, ax=ax, label='FOM')
    rom.mag_plot(w, ax=ax, linestyle='--', label='ROM')
    ax.legend()
    ax.grid()

and plot the magnitude plot of the error system

.. jupyter-execute::

    (fom - rom).mag_plot(w)
    plt.grid()

We can compute the relative errors in :math:`\mathcal{H}_\infty` or
:math:`\mathcal{H}_2` (or Hankel) norm

.. jupyter-execute::

    print(f'Relative Hinf error: {(fom - rom).hinf_norm() / fom.hinf_norm():.3e}')
    print(f'Relative H2 error:   {(fom - rom).h2_norm() / fom.h2_norm():.3e}')

To compute the :math:`\mathcal{H}_\infty` norms, pyMOR uses the dense solver
from Slycot.
