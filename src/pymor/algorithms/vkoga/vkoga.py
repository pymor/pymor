# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults


class VKOGASurrogate(WeakGreedySurrogate):
    """Surrogate for the weak greedy error used when training the :class:`VKOGAEstimator`.

    Not intended to be used directly.
    """

    def __init__(self, kernel, X_train, F_train, criterion='fp', reg=1e-12):
        assert criterion in ('fp', 'f', 'p', 'f/p')
        self.__auto_init(locals())
        self.X_train = np.asarray(X_train)
        self.F_train = np.asarray(F_train)
        if self.F_train.ndim == 1:
            self.F_train = self.F_train.reshape((-1, 1))
        self.N, self.m = self.F_train.shape
        self.d = self.X_train.shape[1]

        self._centers = None
        self._centers_idx = None
        self._coefficients = None
        self._z = None
        self._K_X_mu = None
        self._V = None
        self._C = None

        self._init_power_function_evals()

    def _init_power_function_evals(self):
        """Initialize power2 and V when no centers were selected yet."""
        # power2 = trace(k(x,x)) for each training point
        diag = self.kernel.diag(self.X_train)
        self._power2 = diag

    def _find_training_indices(self, mus):
        """Find mus within training data points (or their nearest neighbors)."""
        # exact matches
        eq = np.isclose(self.X_train[None, :, :], mus[:, None, :]).all(axis=2)
        exact = eq.argmax(axis=1)  # 0 if no match, index otherwise
        has_exact = eq.any(axis=1)

        # nearest neighbors only for non-matches
        if not has_exact.all():
            diff = self.X_train[None, :, :] - mus[:, None, :]
            dists = np.linalg.norm(diff, axis=2)
            nn = dists.argmin(axis=1)
            exact = np.where(has_exact, exact, nn)

        return exact

    def predict(self, X):
        X = np.asarray(X)
        if self._centers is None or self._centers.size == 0:
            return np.zeros((X.shape[0], self.m))

        K = self.kernel(X, self._centers)
        coeff = self._coefficients
        return K @ coeff

    def evaluate(self, mus, return_all_values=False):
        mus = np.asarray(mus)
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

        Incrementally add a new center `mu` with corresponding function value
        from the training data. This function updates

        * the selected centers (`self._centers`)
        * the indices of the selected centers in the training set (`self._centers_idx`)
        * the block-Cholesky factor of the kernel matrix for the selected centers (`self._L`)
        * the coefficients of the interpolant (`self._coefficients`)
        * the power function evaluations on the training set (`self._power2`)
        * the Newton basis evaluations on the training set (`self._V`)
        * the residual :math:`f - s_f` (`self.res`)
        * the inverse of the Cholesky matrix (`self._C`)

        using a numerically stable Newton basis expansion.

        In the following derivation we leave out the regularization term for simplicity.
        If :math:`K_n` denotes the full block kernel for :math:`X_n`, we can write :math:`K_{n+1}`
        as

        .. math::
            K_{n+1} = \begin{bmatrix}
                K_n & B \\
                B^\top & k_{n+1,n+1}
            \end{bmatrix},

        where :math:`B=k(X_n,x_{n+1})` and :math:`k_{n+1,n+1}=k(x_{n+1},x_{n+1})`. Since :math:`K_n`
        is a kernel matrix, it is in particular positive-definite, so it has a Cholesky
        decomposition, i.e. :math:`K_n=L_nL_n^\top`. For :math:`K_{n+1}`, we can compute the
        Cholesky decomposition by a suitable update of :math:`L_n`:

        .. math::
            L_{n+1} = \begin{bmatrix}
                L_n & 0 \\
                W^\top & l_{n+1, n+1}
            \end{bmatrix},

        where :math:`W=L_n^{-1}B` and :math:`l_{n+1, n+1}` is the Cholesky decomposition
        of :math:`k_{n+1,n+1}-W^\top W`.

        It is further possible to update the coefficient vector in a suitable way by reusing
        the old coefficients and without solving the whole system again. To do so, the inverse of
        the Cholesky factor :math:`L_{n}^{-1} =: C_{n}` is required.  We remark that :math:`C_n`
        can also be updated incrementally in a similar fashion as the Cholesky factor.

        Let :math:`c_n\in\mathbb{R}^n` denote the coefficients associated to the interpolant
        for the first :math:`n` selected centers, i.e.

        .. math::
            K_n c_n = f_n,

        where :math:`f_n\in\mathbb{R}^n` corresponds to the vector of target values at the
        selected centers. The interpolant is then given as

        .. math::
            \sum\limits_{i=1}^{n} (c_n)_i k(\,\cdot\,,x_i),

        where :math:`x_1,\dots,x_n\in\mathbb{R}^d` are the :math:`n` selected centers.
        When adding :math:`x_{n+1}\in\mathbb{R}^d` as new center, the new coefficient
        vector :math:`c_{n+1}\in \mathbb{R}^{n+1}` is given as

        .. math::
            c_{n+1} = C_{n+1}^\top \begin{bmatrix}
                c_n \\
                z_{n+1}
            \end{bmatrix}

        for unknown coefficient :math:`z_{n+1}\in\mathbb{R}`, which is computed via the
        residual :math:`r_n` and a scalar version of the power function :math:`p_n`:

        .. math::
            \begin{align*}
               z_{n+1} = {\frac{r_n(x_{n+1})}{p_n(x_{n+1})}
            \end{align*}

        where the residual :math:`r_n` is given by

        .. math::
            \begin{align*}
                r_n(x) := r_{n-1}(x) - z_n V_n(x).
            \end{align*}

        with initial residual :math:`r_0 := f`. Here :math:`V_n` denotes the Newton basis,
        which we define below.  Regarding the power function (measuring for :math:`x\in\mathbb{R}^d`
        how well the current subspace can represent :math:`k(\,\cdot\,,x)`, i.e. the projection
        error in the reproducing kernel Hilbert space):
        The power function :math:`P_n` (for the centers :math:`X_n`) in matrix form is
        defined as

        .. math::
            \begin{align*}
                P_n^2(x) = k(x,x) - k(x,X_n)k(X_n,X_n)^{-1}k(X_n,x).
            \end{align*}

        Since we use the power function to compute the coefficients of the kernel interpolant and
        want to use it as selection criterion for centers within the greedy iteration,
        we consider a scalar version by taking the trace:

        .. math::
            \begin{align*}
                p_n^2(x) = \operatorname{trace}(P_n^2(x)).
            \end{align*}

        We are going to track this quantity evaluated at all training points during the
        iterations of the greedy algorithm. In order to do so, we also maintain an array
        storing the Newton basis evaluations of the current basis at the training points:

        .. math::
            \begin{align*}
                V_{n,i} = C_n k(X_n,y_i)
            \end{align*}

        for all training points :math:`y_i`.

        Given the first center :math:`x_1`, we initialize the (squared) power function
        values as

        .. math::
            \begin{align*}
                p_n^2(y_i) = \operatorname{trace}(k(y_i,y_i)-V_{1,i}^\top V_{1,i}).
            \end{align*}

        The incremental updates of :math:`V_{n,i}` to :math:`V_{n+1,i}` and the power
        function values is then performed in the following way: Compute the latest
        Newton basis via :math:`\Xi_i = C_{n+1}[-m:, :] \cdot k(X_{n+1}, y_i)`

        .. math::
            \begin{align*}
                p_{n+1}^2(y_i) = p_n^2(y_i) - \lVert \Xi_i\rVert_F^2
            \end{align*}

        and update the Newton basis:

        .. math::
            \begin{align*}
                V_{n+1,i} = \begin{bmatrix}
                    V_{n,i} \\
                    \Xi_i
                \end{bmatrix}.
            \end{align*}

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
        self._K_X_mu = self.kernel(self.X_train, mu)

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
        self._power2 = np.maximum(self._power2 - norms, 0.0)


class VKOGAEstimator(BasicObject):
    """Scikit-learn-style estimator using the :class:`VKOGASurrogate`.

    The estimator uses the :func:`~pymor.algorithms.greedy.weak_greedy` in its `fit` method
    to select centers according to the given criterion.

    Parameters
    ----------
    kernel
        Kernel to use in the estimator.
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
        result = weak_greedy(surrogate, X, atol=self.tol, max_extensions=self.max_centers)

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
