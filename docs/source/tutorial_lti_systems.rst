Tutorial: Linear time-invariant systems
=======================================

.. include:: jupyter_init.txt

Heat equation
-------------

We consider the following one-dimensional heat equation over :math:`(0, 1)` with
two inputs :math:`u_1, u_2` and three outputs :math:`y_1, y_2, y_2`:

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
Here, we show how to build one from custom matrices instead of using a
discretizer as in :doc:`tutorial_builtin_discretizer` and the
:meth:`~pymor.models.basic.InstationaryModel.to_lti` of |InstationaryModel|.
In particular, we will use the
:meth:`~pymor.models.iosys.LTIModel.from_matrices` method of |LTIModel|, which
instantiates an |LTIModel| from NumPy or SciPy matrices.

First, we do the necessary imports and some matplotlib style choices.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.sparse as sps
    from pymor.models.iosys import LTIModel
    from pymor.reductors.bt import BTReductor

    plt.rcParams['axes.grid'] = True

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

To visualize the behavior of the `fom`, we can draw its magnitude plot, i.e.,
a visualization of the mapping :math:`\omega \mapsto H(\imath \omega)`, where
:math:`H(s) = C (s E - A)^{-1} B + D` is the transfer function of the system.

.. jupyter-execute::

    w = np.logspace(-2, 8, 50)
    _ = fom.mag_plot(w)

We can also see the Bode plot, which shows the magnitude and phase of the
components of the transfer function.
In particular, :math:`\lvert H_{ij}(\imath \omega) \rvert` is in subplot
:math:`(2 i - 1, j)` and :math:`\arg(H_{ij}(\imath \omega))` is in subplot
:math:`(2 i, j)`.

.. jupyter-execute::

    _ = fom.bode_plot(w)

Plotting the Hankel singular values shows us how well an LTI system can be
approximated by a reduced-order model

.. jupyter-execute::

    hsv = fom.hsv()
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(hsv) + 1), hsv, '.-')
    _ = ax.set_title('Hankel singular values')

As expected for a heat equation, the Hankel singular values decay rapidly.
