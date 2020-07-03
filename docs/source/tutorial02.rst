Tutorial 2: Reducing a heat equation using balanced truncation
==============================================================

.. code-links::
    :timeout: -1


Heat equation
-------------

We consider the following heat equation over :math:`[0, 1]` with two inputs and
three outputs:

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

In the following, we will create a discretized model and reduced it using the
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

There are many ways of building an |LTIModel|.
Here, we will use its :meth:`~pymor.models.iosys.LTIModel.from_matrices` method,
which instantiates an |LTIModel| from NumPy or SciPy matrices.

First, we do the necessary imports.

.. nbplot::

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sps
    from pymor.basic import LTIModel, BTReductor

Next, we can assemble the matrices:

.. nbplot::

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

.. nbplot::

    fom = LTIModel.from_matrices(A, B, C)

We can get the internal representation of the |LTIModel| `fom`

.. nbplot::

    fom

and some basic information from its string representation

.. nbplot::

    print(fom)

The magnitude plot is

.. nbplot::

    w = np.logspace(-2, 7, 50)
    _ = fom.mag_plot(w)

The Hankel singular values are

.. nbplot::

    hsv = fom.hsv()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    _ = ax.set_title('Hankel singular values')


Running balanced truncation
---------------------------

First, we need the reductor object

.. nbplot::

    bt = BTReductor(fom)

We can use it to compute the a priori :math:`\mathcal{H}_\infty`-error bounds
based on the Hankel singular values.

.. nbplot::

    error_bounds = bt.error_bounds()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(error_bounds) + 1), error_bounds, '.-')
    ax.semilogy(range(1, len(hsv)), hsv[1:], '.-')
    _ = ax.set_title(r'Upper and lower $\mathcal{H}_\infty$-error bounds')

To get a reduced-order model, we call the `reduce` method.

.. nbplot::

    rom = bt.reduce(10)

We can compare the magnitude plots

.. nbplot::

    fig, ax = plt.subplots()
    fom.mag_plot(w, ax=ax)
    _ = rom.mag_plot(w, ax=ax, linestyle='--')

and plot the magnitude plot of the error system

.. nbplot::

    _ = (fom - rom).mag_plot(w)

We can compute the relative errors

.. nbplot::

    print(f'Relative H2-error:   {(fom - rom).h2_norm() / fom.h2_norm():.3e}')
    print(f'Relative Hinf-error: {(fom - rom).hinf_norm() / fom.hinf_norm():.3e}')
