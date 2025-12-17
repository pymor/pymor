# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults


class VKOGAEstimator(BasicObject):
    """Scikit-learn-style estimator using the :class:`VKOGASurrogate`.

    The estimator uses the :func:`~pymor.algorithms.greedy.weak_greedy` in its `fit`-method
    to select centers according to the given criterion.

    The algorithm is described in :cite:`WH13` and :cite:`SH21`.

    Parameters
    ----------
    kernel
        Kernel to use in the estimator. The kernel is assumed to have a scalar-valued output.
        For vector-valued outputs, the interpolant uses vector-valued coefficients,
        i.e., the prediction is computed as a linear combination of kernel evaluations
        with vector-valued weights. The interface of the kernel needs to follow the scikit-learn
        interface and in particular a `__call__`-method for (vectorized) evaluation of the kernel
        and a `diag`-method for computing the diagonal of the kernel matrix are required.
        For convenience, a Gaussian kernel is provided in :mod:`pymor.algorithms.vkoga.kernels`.
    criterion
        Selection criterion for the greedy algorithm. Possible values are `'fp'`, `'f'` and `'p'`.
    max_centers
        Maximum number of selected centers in the greedy algorithm.
    tol
        Tolerance for the weak greedy algorithm.
    reg
        Regularization parameter for the kernel interpolation.
    """

    @defaults('criterion', 'max_centers', 'tol', 'reg')
    def __init__(self, kernel, criterion='fp', max_centers=20, tol=1e-6, reg=1e-12):
        self.__auto_init(locals())
        self._surrogate = None

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
        The trained estimator.
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # instantiate surrogate
        surrogate = VKOGASurrogate(kernel=self.kernel, X_train=X, F_train=Y, criterion=self.criterion, reg=self.reg)

        # use X as training set in the weak greedy algorithm
        result = weak_greedy(surrogate, np.arange(len(X)), atol=self.tol, max_extensions=self.max_centers)

        self._surrogate = surrogate
        # store the results of the weak greedy algorithm for inspection/plotting
        self._weak_greedy_result = result

        return self

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
        if self._surrogate is None:
            raise RuntimeError('Call fit() before predict().')
        return self._surrogate.predict(X)


class VKOGASurrogate(WeakGreedySurrogate):
    """Surrogate for the weak greedy error used when training the :class:`VKOGAEstimator`.

    Not intended to be used directly.
    """

    def __init__(self, kernel, X_train, F_train, criterion='fp', reg=1e-12):
        assert criterion in ('fp', 'f', 'p')
        X_train = np.asarray(X_train)
        F_train = np.asarray(F_train)
        if F_train.ndim == 1:
            F_train = F_train.reshape((-1, 1))
        self.__auto_init(locals())
        self.N, self.m = self.F_train.shape
        self.d = self.X_train.shape[1]

        self._centers = None
        self._centers_idx = None
        self._coefficients = None
        self._z = None
        self._K_X_mu = None
        self._V = None
        self._C = None
        # power2 = trace(k(x,x)) for each training point
        self._power2 = self.kernel.diag(self.X_train) + self.reg

    def _find_training_indices(self, mus):
        """Find nearest neighbors of mus within training data points."""
        diff = self.X_train[None, :, :] - mus[:, None, :]
        dists = np.linalg.norm(diff, axis=2)
        return dists.argmin(axis=1)

    def predict(self, X):
        X = np.asarray(X)
        if self._centers is None:
            return np.zeros((X.shape[0], self.m))

        K = self.kernel(X, self._centers)
        coeff = self._coefficients
        return K @ coeff

    def evaluate(self, mus, return_all_values=False):
        mus = np.asarray(self.X_train[mus])
        idxs = self._find_training_indices(mus)
        Xc = mus
        p2 = self._power2[idxs]

        if self.criterion in ('fp', 'f'):
            Ypred = self.predict(Xc)
            Fres = self.F_train[idxs] - Ypred
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
        return scores[idx], mus[idx]

    def extend(self, mu):
        r"""Extends the kernel interpolant by adding a new center and updating all quantities.

        Incrementally add a new center :math:`\mu` with corresponding function value
        from the training data. This function updates

        * the selected centers (`self._centers`)
        * the indices of the selected centers in the training set (`self._centers_idx`)
        * the coefficients of the interpolant (`self._coefficients`)
        * the (squared) power function evaluations on the training set (`self._power2`)
        * the Newton basis evaluations on the training set (`self._V`)
        * the residual :math:`f - s_f` at the training points (`self.res`)
        * the inverse of the Cholesky factor of the kernel matrix (`self._C`)
        * the coefficients in the Newton basis (`self._z`).

        In the following derivation we leave out the regularization term for simplicity
        and restrict ourselves to scalar kernels.
        Let :math:`K_n` denote the full kernel matrix for the current set of selected
        centers :math:`X_n`. Since :math:`K_n` is a kernel matrix, it is in particular
        positive-definite, so it has a Cholesky decomposition,
        i.e. :math:`K_n=L_nL_n^\top`. The inverse of the Choleksy decomposition
        :math:`C_n := L_{n}^{-1}` will be used to efficiently compute and update the
        coefficients  of the kernel interpolant. The formula for the computation and update of
        :math:`C_n` can be found below.

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
        :math:`X_{n+1}=X_n\cup \{\mu\}`, the new coefficient
        vector :math:`c_{n+1}\in\mathbb{R}^{n+1}` is given as

        .. math::
            c_{n+1} = C_{n+1}^\top \begin{bmatrix}
                c_n \\
                z_{n+1}
            \end{bmatrix}

        for an unknown coefficient :math:`z_{n+1}\in\mathbb{R}`, which is computed via the
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

        It remains the computation of the inverse of the Cholesky factor :math:`C_n`:
        The initial inverse Cholesky factor is given as

        .. math::
            \begin{align*}
                C_1 = \frac{1}{V_1(\mu)}
            \end{align*}

        and is updated via

        .. math::
            C_{n+1} = \begin{bmatrix}
                C_n & 0 \\
                -\frac{1}{V_{n+1}(\mu)}\bar{v}_{n}(\mu) C_n & \frac{1}{V_{n+1}(\mu)}
            \end{bmatrix},

        where we denote as :math:`\bar{v}_{n}(\mu)\in\mathbb{R}^n`
        the vector :math:`[V_{1}(\mu),\ldots,V_{n}(\mu)]`.

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

        # update Cholesky inverse, Newton basis and the power function values
        self._update_newton_basis(idx_in_X)
        self._update_cholesky_inverse(idx_in_X)
        self._update_power_function_evals()

        # update centers and final coefficients
        self._coefficients = self._C.T @ self._z
        if self._centers is None and self._centers_idx is None:
            self._centers = np.atleast_2d(mu)
            self._centers_idx = np.array([idx_in_X], dtype=int)
        else:
            self._centers = np.vstack([self._centers, mu])
            self._centers_idx = np.concatenate([self._centers_idx, np.array([idx_in_X], dtype=int)])

    def _update_newton_basis(self, idx_in_X):
        """Append Xi as the last block to the Newton basis V."""
        if self._V is None:
            self._V = self._K_X_mu / np.sqrt(self._power2[idx_in_X])
        else:
            Xi = self._K_X_mu - (self._V[idx_in_X] @ self._V.T).reshape(-1, 1)
            Xi = Xi / np.sqrt(self._power2[idx_in_X])
            self._V = np.concatenate([self._V, Xi], axis=1)

    def _update_cholesky_inverse(self, idx_in_X):
        """Extend the inverse of the Cholesky matrix by the new block."""
        c_nn = 1 / self._V[idx_in_X, -1]
        if self._C is None:
            self._C = np.atleast_2d(c_nn)
        else:
            self._C = np.block([[self._C, np.zeros((self._C.shape[0], 1))],
                                [- c_nn * self._V[idx_in_X, :-1] @ self._C, c_nn]])

    def _update_power_function_evals(self):
        """Incrementally update self._power2 after adding new center."""
        # update power2: p_{n+1}^2(x) = p_n^2(x) - norms
        norms = self._V[:, -1] ** 2
        self._power2 = np.maximum(self._power2 - norms, 0)
