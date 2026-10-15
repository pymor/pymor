# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.linalg import solve_triangular

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.algorithms.ml.base_regressor import BaseRegressor
from pymor.algorithms.ml.vkoga.kernels import GaussianKernel
from pymor.core.defaults import defaults
from pymor.core.exceptions import ExtensionError


class VKOGARegressor(BaseRegressor):
    """Scikit-learn-style regressor using the :class:`VKOGASurrogate`.

    The regressor uses the :func:`~pymor.algorithms.greedy.weak_greedy` in its `fit`-method
    to select centers according to the given criterion.

    The algorithm is described in :cite:`WH13` and :cite:`SH21`.

    Parameters
    ----------
    kernel
        Kernel to use in the regressor. The kernel is assumed to have a scalar-valued output.
        For vector-valued outputs, the interpolant uses vector-valued coefficients,
        i.e., the prediction is computed as a linear combination of kernel evaluations
        with vector-valued weights. The interface of the kernel needs to follow the scikit-learn
        interface and in particular a `__call__`-method for (vectorized) evaluation of the kernel
        and a `diag`-method for computing the diagonal of the kernel matrix are required.
        For convenience, a Gaussian kernel is provided in :mod:`pymor.algorithms.ml.vkoga.kernels`.
        If `kernel=None` is passed, a Gaussian kernel will be initialized by default.
    criterion
        Selection criterion for the greedy algorithm. Possible values are `'fp'`, `'f'` and `'p'`.
    max_centers
        Maximum number of selected centers in the greedy algorithm.
    tol
        Tolerance for the weak greedy algorithm.
    reg
        Regularization parameter for the kernel interpolation.
    """

    _params = ('kernel', 'criterion', 'max_centers', 'tol', 'reg')
    _nested_object = 'kernel'

    @defaults('kernel', 'criterion', 'max_centers', 'tol', 'reg')
    def __init__(self, kernel=None, criterion='fp', max_centers=20, tol=1e-6, reg=1e-12):
        if kernel is None:
            kernel = GaussianKernel()

        self.kernel = kernel
        self.criterion = criterion
        self.max_centers = max_centers
        self.tol = tol
        self.reg = reg

    def fit(self, X, Y):
        """Fit VKOGA surrogate using pyMOR's weak greedy algorithm.

        Parameters
        ----------
        X
            Training inputs.
        Y
            Training targets.

        Returns
        -------
        The trained regressor.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # instantiate surrogate
        surrogate = VKOGASurrogate(kernel=self.kernel, X_train=X, F_train=Y, criterion=self.criterion, reg=self.reg)

        # use X as training set in the weak greedy algorithm
        result = weak_greedy(surrogate, np.arange(len(X)), atol=self.tol, max_extensions=min(len(X), self.max_centers))

        self._surrogate = surrogate
        # store the results of the weak greedy algorithm for inspection/plotting
        self._weak_greedy_result = result

        return self

    def extend(self, X, Y):
        """Incrementally extend the VKOGA surrogate with new training data.

        New data points are added to the surrogate's training set. A greedy
        step is then performed to decide whether any of the new points should
        be selected as centers.

        Parameters
        ----------
        X
            New training inputs.
        Y
            New training targets.
        """
        if self._surrogate is None:
            raise RuntimeError('Call fit() before extend().')
        X = np.asarray(X)
        Y = np.asarray(Y)
        self._surrogate.extend_training_data(X, Y)

        # run greedy on the new points to decide if they should become centers
        new_indices = np.arange(self._surrogate.N - len(X), self._surrogate.N)
        n_existing = len(self._surrogate._centers) if self._surrogate._centers is not None else 0
        remaining_centers = self.max_centers - n_existing
        if remaining_centers > 0:
            weak_greedy(self._surrogate, new_indices, atol=self.tol, max_extensions=remaining_centers)

    def predict(self, X):
        """Predict the target for the input `X`.

        Parameters
        ----------
        X
            Input for which to compute the prediction.

        Returns
        -------
        Prediction obtained by the :class:`VKOGASurrogate`.
        """
        if not hasattr(self, '_surrogate'):
            raise RuntimeError('Call fit() before predict().')
        return self._surrogate.predict(X)


class VKOGASurrogate(WeakGreedySurrogate):
    """Surrogate for the weak greedy error used when training the :class:`VKOGARegressor`.

    Not intended to be used directly.
    """

    def __init__(self, kernel, X_train, F_train, criterion='fp', reg=1e-12):
        assert criterion in ('fp', 'f', 'p')
        X_train = np.asarray(X_train)
        F_train = np.asarray(F_train)
        if F_train.ndim == 1:
            F_train = F_train.reshape((-1, 1))

        self.kernel = kernel
        self.X_train = X_train
        self.F_train = F_train
        self.criterion = criterion
        self.reg = reg

        self.N, self.m = self.F_train.shape
        self.d = self.X_train.shape[1]

        self._centers = None
        self._centers_idx = None
        self._z = None
        self._K_X_mu = None
        self._V = None
        self._V_buf = None
        self.res = None
        # power2 = trace(k(x,x)) for each training point
        self._power2 = self.kernel.diag(self.X_train) + self.reg

    def extend_training_data(self, X_new, F_new):
        """Add new training points and update all internal quantities.

        Parameters
        ----------
        X_new
            New training inputs.
        F_new
            New training targets.
        """
        X_new = np.atleast_2d(X_new)
        F_new = np.asarray(F_new)
        if F_new.ndim == 1:
            F_new = F_new.reshape((-1, 1))

        n_new = X_new.shape[0]
        assert X_new.shape[1] == self.d
        assert F_new.shape[1] == self.m

        # extend training data arrays
        self.X_train = np.vstack([self.X_train, X_new])
        self.F_train = np.vstack([self.F_train, F_new])
        self.N += n_new

        # extend power function evaluations for new points
        new_power2 = self.kernel.diag(X_new) + self.reg

        if self._V is not None:
            n_bases = self._V.shape[1]
            V_new = np.zeros((n_new, n_bases))
            for j in range(n_bases):
                center_idx = self._centers_idx[j]
                K_new_center = self.kernel(X_new, self.X_train[center_idx:center_idx+1]).ravel()
                V_new[:, j] = (K_new_center - V_new[:, :j] @ self._V[center_idx, :j]) / self._V[center_idx, j]
                new_power2 -= V_new[:, j] ** 2
            self._append_newton_rows(V_new)

            # residual must match the state of old points: F - V[:, :-1] @ z[:-1]
            new_res = F_new - V_new[:, :-1] @ self._z[:-1]
            self.res = np.vstack([self.res, new_res])
        else:
            # no centers yet, residual is just the raw target values
            if self.res is not None:
                self.res = np.vstack([self.res, F_new])
            else:
                self.res = F_new

        self._power2 = np.concatenate([self._power2, np.maximum(new_power2, 0)])

    def predict(self, X):
        X = np.asarray(X)
        if self._centers is None:
            return np.zeros((X.shape[0], self.m))

        K = self.kernel(X, self._centers)
        V_selected = self._V[self._centers_idx, :]
        coeff = solve_triangular(V_selected.T, self._z, lower=False)
        return K @ coeff

    def evaluate(self, mus, return_all_values=False):
        mus = np.asarray(mus)
        idxs = mus
        Xc = self.X_train[idxs]
        p2 = self._power2[idxs]

        if self.criterion in ('fp', 'f'):
            if self.res is None and self._V is None and self._z is None:
                Fres = self.F_train[idxs]
            else:
                Fres = (self.res - self._V[:, -1:] @ self._z[-1:])[idxs]
            res_norms = np.linalg.norm(Fres, axis=1)
            if self.criterion == 'fp':
                scores = res_norms * np.sqrt(p2)
            elif self.criterion == 'f':
                scores = res_norms
        elif self.criterion == 'p':
            scores = np.sqrt(p2)

        if return_all_values:
            return scores
        idx = np.argmax(scores)
        return scores[idx], Xc[idx]

    def extend(self, mu):
        r"""Extends the kernel interpolant by adding a new center and updating all quantities.

        Incrementally add a new center :math:`\mu` with corresponding function value
        from the training data. This function updates

        * the selected centers (`self._centers`)
        * the indices of the selected centers in the training set (`self._centers_idx`)
        * the (squared) power function evaluations on the training set (`self._power2`)
        * the Newton basis evaluations on the training set (`self._V`)
        * the residual :math:`f - s_f` at the training points (`self.res`)
        * the coefficients in the Newton basis (`self._z`).

        In the following derivation we leave out the regularization term for simplicity
        and restrict ourselves to scalar kernels.
        Let :math:`K_n` denote the full kernel matrix for the current set of selected
        centers :math:`X_n`. Since :math:`K_n` is a kernel matrix, it is in particular
        positive-definite, so it has a Cholesky decomposition,
        i.e. :math:`K_n=L_nL_n^\top`. Evaluating the Newton basis at the selected centers
        yields a lower-triangular matrix that equals :math:`L_n`. Rather than tracking
        :math:`L_n` explicitly, the coefficients of the kernel interpolant are recovered
        from the Newton-basis coefficients by a single triangular solve at prediction time
        (see :meth:`predict`).

        Updating the coefficients of the kernel interpolant: Let :math:`c_n\in\mathbb{R}^n` denote
        the coefficients associated to the kernel interpolant for the first :math:`n` selected
        centers (:math:`X_n`), i.e.

        .. math::
            K_n c_n = f_n,

        where :math:`f_n\in\mathbb{R}^n` corresponds to the vector of target values at the
        selected centers. The kernel interpolant is then given as

        .. math::
            s_f^n(x) := \sum\limits_{i=1}^{n} (c_n)_i k(x, x_i),

        where :math:`X_n=\{x_1,\dots,x_n\}\subset\mathbb{R}^d` is the set containing the :math:`n`
        selected centers. When adding :math:`\mu` as new center, i.e.
        :math:`X_{n+1}=X_n\cup \{\mu\}`, the updated kernel coefficients
        :math:`c_{n+1}\in\mathbb{R}^{n+1}` are recovered from the Newton-basis
        coefficients :math:`z_1,\ldots,z_{n+1}` by solving

        .. math::
            L_{n+1}^\top c_{n+1} =
                \begin{bmatrix} z_1 & \cdots & z_{n+1} \end{bmatrix}^\top.

        The new Newton-basis coefficient :math:`z_{n+1}\in\mathbb{R}` is computed via the
        residual :math:`r_n` and the power function :math:`P_n` for the centers :math:`X_n`
        evaluated at :math:`\mu` (the definitions of these quantities are given below):

        .. math::
            \begin{align*}
               z_{n+1} = \frac{r_n(\mu)}{P_n(\mu)}.
            \end{align*}

        The residual :math:`r_n=f-s_f^n` is defined as the difference between the target function
        :math:`f` and the current kernel interpolant :math:`s_f^n` and can be updated via
        the following recursion:

        .. math::
            \begin{align*}
                r_n(x) = r_{n-1}(x) - z_n V_n(x)
            \end{align*}

        with initial residual :math:`r_0 = f`. Here, :math:`V_n` denotes the Newton basis for

        .. math::
            \begin{align*}
                V(X_n) := \mathrm{span}\{k(\,\cdot\,,x_1), \dots, k(\,\cdot\,,x_n)\}.
            \end{align*}

        The first Newton basis is given as

        .. math::
            \begin{align*}
                V_{1}(x)= \frac{k(x, x_1)}{\sqrt{k(x_1, x_1)}}.
            \end{align*}

        The subsequent Newton basis functions are computed via the Gram-Schmidt algorithm as

        .. math::
            \begin{align*}
                V_{n+1}(x) = \frac{k(x, \mu) - \sum_{j=1}^n V_j(x) V_j(\mu)}{P_n(\mu)}.
            \end{align*}

        Regarding the power function (measuring for  :math:`x\in\mathbb{R}^d` how well the current
        subspace can represent  :math:`k(\,\cdot\,,x)`, i.e. the projection error in the
        reproducing kernel  Hilbert space): The initial squared power function is given as

        .. math::
            \begin{align*}
                P_0^2(x) = k(x, x)
            \end{align*}

        and the incremental update of the squared power function :math:`P_n^2` can be computed by

        .. math::
            \begin{align*}
                P_n^2(x) = P_{n-1}^2(x) - V_n(x)^2.
            \end{align*}

        We are going to track this quantity evaluated at all training points during the
        iterations of the greedy algorithm.

        Parameters
        ----------
        mu
            Parameter (center) to add.
        """
        mu = np.asarray(mu)

        # --- Find corresponding training index (exact or nearest match) ---
        # this is mainly required since we do not enforce that `mu` is in the training set;
        # when introducing a suitable `Dataset` class, this should be handled by the dataset
        matches = np.where(np.all(np.isclose(self.X_train, mu), axis=1))[0]
        if matches.size > 0:
            idx_in_X = int(matches[0])
        else:
            idx_in_X = int(np.argmin(np.linalg.norm(self.X_train - mu, axis=1)))

        if self._centers_idx is not None and idx_in_X in self._centers_idx:
            raise ExtensionError('Center already selected.')

        if self._power2[idx_in_X] <= 0 or np.isclose(self._power2[idx_in_X], 0):
            raise ExtensionError('Power function value is zero for selected center.')

        # update the residual
        if self._V is None and self._z is None:
            self.res = self.F_train
        else:
            self.res = self.res - self._V[:, -1:] @ self._z[-1:]

        self._extend_incremental(mu, idx_in_X)

    def _extend_incremental(self, mu, idx_in_X):
        """Incrementally add a new center to the existing interpolant."""
        # compute k(yi, mu) for all training points y_i
        self._K_X_mu = self.kernel(self.X_train, np.atleast_2d(mu))

        # update coefficient
        z_new = (self.res[idx_in_X] / np.sqrt(self._power2[idx_in_X])).reshape(1, self.m)
        self._z = np.vstack([self._z, z_new]) if self._z is not None else z_new

        # update Newton basis and the power function values
        self._update_newton_basis(idx_in_X)
        self._update_power_function_evals()

        # update centers
        if self._centers is None and self._centers_idx is None:
            self._centers = np.atleast_2d(mu)
            self._centers_idx = np.array([idx_in_X], dtype=int)
        else:
            self._centers = np.vstack([self._centers, mu])
            self._centers_idx = np.concatenate([self._centers_idx, np.array([idx_in_X], dtype=int)])

    def _update_newton_basis(self, idx_in_X):
        """Append Xi as the last block to the Newton basis V."""
        if self._V is None:
            Xi = self._K_X_mu / np.sqrt(self._power2[idx_in_X])
        else:
            Xi = self._K_X_mu - (self._V[idx_in_X] @ self._V.T).reshape(-1, 1)
            Xi = Xi / np.sqrt(self._power2[idx_in_X])
        self._append_newton_column(Xi)

    def _append_newton_column(self, Xi):
        """Append Xi (shape ``(N, 1)``) as a new column of V.

        ``self._V`` stays an ``(N, n)`` view into a preallocated buffer, so all
        existing indexing (``-1``, ``.shape[1]``, row/column slices) keeps
        working; the buffer only grows (by doubling) when it runs out of columns.
        """
        n = 0 if self._V is None else self._V.shape[1]
        if self._V_buf is None or self._V_buf.shape[1] < n + 1:
            cap = max(2 * (self._V_buf.shape[1] if self._V_buf is not None else 0), 8)
            buf = np.empty((self.X_train.shape[0], cap))
            if self._V is not None:
                buf[:, :n] = self._V
            self._V_buf = buf
        self._V_buf[:, n] = Xi.ravel()
        self._V = self._V_buf[:, :n + 1]

    def _append_newton_rows(self, V_new):
        """Append V_new (shape ``(n_new, n)``) as new training rows of V."""
        if self._V_buf is None:
            self._V_buf = np.ascontiguousarray(self._V)
        n = self._V.shape[1]
        old_n = self._V.shape[0]
        buf = np.empty((old_n + V_new.shape[0], self._V_buf.shape[1]))
        buf[:old_n, :n] = self._V
        buf[old_n:, :n] = V_new
        self._V_buf = buf
        self._V = buf[:, :n]

    def _update_power_function_evals(self):
        """Incrementally update self._power2 after adding new center."""
        # update power2: p_{n+1}^2(x) = p_n^2(x) - norms
        norms = self._V[:, -1] ** 2
        self._power2 = np.maximum(self._power2 - norms, 0)
