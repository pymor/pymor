Tutorial: Reducing a heat equation using balanced truncation
============================================================

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


Running balanced truncation
---------------------------

The balanced truncation method consists of finding a balanced realization of the
full-order LTI system and truncating it to obtain a reduced-order model. In
particular, there exist invertible transformation matrices
:math:`T, S \in \mathbb{R}^{n \times n}` such that the equivalent full-order
model with
:math:`\widetilde{E} = S^T E T = I`,
:math:`\widetilde{A} = S^T A T`,
:math:`\widetilde{B} = S^T B`,
:math:`\widetilde{C} = C T`
has Gramians :math:`\widetilde{P}` and :math:`\widetilde{Q}`, i.e., solutions to
Lyapunov equations

.. math::

    \begin{align}
        \widetilde{A} \widetilde{P}
        + \widetilde{P} \widetilde{A}^T
        + \widetilde{B} \widetilde{B}^T
        & = 0, \\
        \widetilde{A}^T \widetilde{Q}
        + \widetilde{Q} \widetilde{A}
        + \widetilde{C}^T \widetilde{C}
        & = 0, \\
    \end{align}

such that
:math:`\widetilde{P} = \widetilde{Q} = \Sigma = \operatorname{diag}(\sigma_i)`,
where :math:`\sigma_i` are the Hankel singular values.
Then, taking as basis matrices :math:`V, W \in \mathbb{R}^{n \times r}` the
first :math:`r` columns of :math:`T` and :math:`S` (possibly after
orthonormalization), gives a reduced-order model

.. math::

    \begin{align}
        \widehat{E} \dot{\widehat{x}}(t)
        & = \widehat{A} \widehat{x}(t) + \widehat{B} u(t), \\
        \widehat{y}(t)
        & = \widehat{C} \widehat{x}(t) + D u(t),
    \end{align}

with
:math:`\widehat{E} = W^T E V`,
:math:`\widehat{A} = W^T A V`,
:math:`\widehat{B} = W^T B`,
:math:`\widehat{C} = C V`,
which satisfies the :math:`\mathcal{H}_\infty` (i.e., induced
:math:`\mathcal{L}_2`) error bound

.. math::

    \sup_{u \neq 0} \frac{\lVert y - \widehat{y} \rVert_{\mathcal{L}_2}}{\lVert u \rVert_{\mathcal{L}_2}}
    \leqslant 2 \sum_{i = r + 1}^n \sigma_i.

Note that any reduced-order model (not only from balanced truncation) satisfies
the lower bound

.. math::

    \sup_{u \neq 0} \frac{\lVert y - \widehat{y} \rVert_{\mathcal{L}_2}}{\lVert u \rVert_{\mathcal{L}_2}}
    \geqslant \sigma_{r + 1}.

To run balanced truncation in pyMOR, we first need the reductor object

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
    _ = ax.set_title(r'Upper and lower $\mathcal{H}_\infty$ error bounds')

To get a reduced-order model of order 10, we call the `reduce` method with the
appropriate argument:

.. jupyter-execute::

    rom = bt.reduce(10)

Instead, or in addition, a tolerance for the :math:`\mathcal{H}_\infty` error
can be specified, as well as the projection algorithm (by default, the
balancing-free square root method is used).
The used Petrov-Galerkin bases are stored in `bt.V` and `bt.W`.

We can compare the magnitude plots between the full-order and reduced-order
models

.. jupyter-execute::

    fig, ax = plt.subplots()
    fom.mag_plot(w, ax=ax, label='FOM')
    rom.mag_plot(w, ax=ax, linestyle='--', label='ROM')
    _ = ax.legend()

as well as Bode plots

.. jupyter-execute::

    fig, axs = plt.subplots(6, 2, figsize=(12, 24), sharex=True, constrained_layout=True)
    fom.bode_plot(w, ax=axs)
    _ = rom.bode_plot(w, ax=axs, linestyle='--')

Also, we can plot the magnitude plot of the error system

.. jupyter-execute::

    err = fom - rom
    _ = err.mag_plot(w)

and its Bode plot

.. jupyter-execute::

    _ = err.bode_plot(w)

We can compute the relative errors in :math:`\mathcal{H}_\infty` or
:math:`\mathcal{H}_2` (or Hankel) norm

.. jupyter-execute::

    print(f'Relative Hinf error: {err.hinf_norm() / fom.hinf_norm():.3e}')
    print(f'Relative H2 error:   {err.h2_norm() / fom.h2_norm():.3e}')

.. note::

    To compute the :math:`\mathcal{H}_\infty` norms, pyMOR uses the dense solver
    from Slycot, and therefore all of the operators have to be converted to
    dense matrices. For large systems, this may be very expensive.

Download the code:
:jupyter-download:script:`tutorial_bt`
:jupyter-download:notebook:`tutorial_bt`
