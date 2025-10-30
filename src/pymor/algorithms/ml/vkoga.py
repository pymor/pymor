# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.linalg import solve_triangular

from pymor.algorithms.greedy import WeakGreedySurrogate, weak_greedy
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults


class VKOGASurrogate(WeakGreedySurrogate):
    """Surrogate for the weak greedy error used when training the :class:`VKOGAEstimator`.

    Not intended to be used directly.
    """

    def __init__(self, kernel, X_train, F_train, criterion='fp', reg=1e-12):
        assert criterion in ('fp', 'f', 'p')
        self.__auto_init(locals())
        self.X_train = np.asarray(X_train)
        self.F_train = np.asarray(F_train)
        self.N, self.m = self.F_train.shape

        self._centers = None
        self._centers_idx = None
        self._coefficients = None
        self.L = None  # block-Cholesky factor of the kernel matrix

    def _stack_block_column(self, X_centers, x_new):
        """Stack K(c_j, x_new) for j over X_centers in order. Result shape (n*m, m)."""
        if X_centers is None:
            return np.zeros((0, self.m))
        n = len(X_centers)
        B = np.zeros((n * self.m, self.m))
        for j, cj in enumerate(X_centers):
            B[j*self.m:(j+1)*self.m, :] = self.kernel(cj, x_new)
        return B

    def predict(self, X):
        X = np.asarray(X)
        if self._centers is None:
            return np.zeros((X.shape[0], self.m))
        y = np.zeros((X.shape[0], self.m))
        for j, cj in enumerate(self._centers):
            K_vals = np.array([self.kernel(x, cj) for x in X])
            y += np.einsum('nij,j->ni', K_vals, self._coefficients[j])
        return y

    def power_function(self, X):
        X = np.asarray(X)
        if self.L is None:
            return np.ones(X.shape[0])
        P = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            B_col = self._stack_block_column(self._centers, x)
            K_xx = self.kernel(x, x)
            v = solve_triangular(self.L, B_col, lower=True)
            S = K_xx - v.T @ v
            P[i] = np.sqrt(max(0.0, np.linalg.norm(S, ord='fro')))
        return P

    def evaluate(self, mus, return_all_values=False):
        mus = np.asarray(mus)
        # map mus to exact training indices if possible, else nearest
        idxs = []
        for mu in mus:
            matches = np.where(np.all(np.isclose(self.X_train, mu), axis=1))[0]
            if matches.size > 0:
                idxs.append(int(matches[0]))
            else:
                # fallback to nearest but warn once
                idx_nn = int(np.argmin(np.linalg.norm(self.X_train - mu, axis=1)))
                idxs.append(idx_nn)
                self.logger.warn('Parameter not in the training set, falling back to nearest.')

        idxs = np.array(idxs, dtype=int)
        Xc = mus

        if self.criterion in ('fp', 'f'):
            Ypred = self.predict(Xc)
            Fres = self.F_train[idxs] - Ypred
            res_norms = np.linalg.norm(Fres, axis=1)
            if self.criterion == 'fp':
                P = self.power_function(Xc)
                scores = res_norms * P
            elif self.criterion == 'f':
                scores = res_norms
        elif self.criterion == 'p':
            P = self.power_function(Xc)
            scores = P

        if return_all_values:
            return scores
        idx = int(np.argmax(scores))
        return float(scores[idx]), mus[idx]

    def extend(self, mu):
        r"""Extends the kernel interpolant by adding a new center and updating all quantities.

        Incrementally add a new center `mu` with corresponding function value
        from the training data. This function updates

        * _centers
        * _centers_idx
        * L (block-Cholesky factor)
        * _coefficients

        using a numerically stable Newton-basis expansion.

        In the following derivation we leave out the regularization term for simplicity.
        If :math:`K_n` denotes the full block kernel matrix, we can write :math:`K_{n+1}` as

        .. math::
            K_{n+1} = \begin{bmatrix}
                K_n & B \\
                B^\top K_{nn}
            \end{bmatrix},

        where :math:`B=k(X_n,x_{n+1})` and :math:`K_{nn}=k(x_{n+1},x_{n+1})`. Since :math:`K_n`
        is a kernel matrix, it is in particular positive-definite, so it has a Cholesky
        decomposition, i.e. :math:`K_n=L_nL_n^\top`. For :math:`K_{n+1}`, we can compute the
        Cholesky decomposition by a suitable update of :math:`L_n`:

        .. math::
            L_{n+1} = \begin{bmatrix}
                L_n & 0 \\
                W^\top & L_{22}
            \end{bmatrix},

        where :math:`W=L_n^{-1}B` and :math:`L_{22}` is the Cholesky decomposition
        of :math:`K_{nn}-W^\top W`.

        It is further possible to update the coefficient vector in a suitable way by reusing
        the old coefficients and without solving the whole system again:
        Let :math:`c_n\in\mathbb{R}^n` denote the coefficients associated to the interpolant
        for the first :math:`n` selected centers, i.e.

        .. math::
            K_n c_n = f_n,

        where :math:`f_n\in\mathbb{R}^n` corresponds to the vector of target values at the
        selected centers. The interpolant is then given as

        .. math::
            \sum\limits_{i=1}^{n} (c_n)_i k(\,\cdot\,,x_i),

        where :math:`x_1,\dots,x_n\in\mathbb{R}^d` are the :math:`n` selected centers.
        When adding :math:`x_{n+1}\in\mathbb{R}^d` as new center, we can compute
        :math:`c_{n+1}\in\mathbb{R}^{n+1}` in the following way: Let us write

        .. math::
            c_{n+1} = \begin{bmatrix}
                a_1 \\
                a_2
            \end{bmatrix}

        for unknown coefficients :math:`a_1\in\mathbb{R}^n` and :math:`a_2\in\mathbb{R}`.
        We then have (by splitting the right-hand side :math:`f_{n+1}` accordingly)

        .. math::
            \begin{align*}
                K_na_1+Ba_2 &= f_1, \\
                B^\top a_1+K_{nn}a_2 &= f_2.
            \end{align*}

        One can now solve for :math:`a_1` and :math:`a_2` in the following way:

        .. math::
            \begin{align*}
                y_1 &= L_n^{-1} f_1,\\
                y_2 &= L_{22}^{-1}(f_2-W^\top y_1),\\
                a_1 &= L_n^{-\top}(y_1-Wy_2) = c_n - L_n^{-\top}Wy_2,\\
                a_2 &= L_{22}^{-\top}y_2,
            \end{align*}

        where we observe that :math:`y_1` is given by the last iteration and therefore
        the update of :math:`a_1` and the computation of :math:`a_2` are
        relatively cheap.

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

        f_new = self.F_train[idx_in_X].reshape(self.m,)

        # --- Case 1: first center ---
        if self.L is None or self._centers is None:
            K_nn = self.kernel(mu, mu) + self.reg * np.eye(self.m)
            L22 = np.linalg.cholesky(K_nn)
            y_new = solve_triangular(L22, f_new, lower=True)
            a2 = solve_triangular(L22.T, y_new, lower=False)
            self.L = L22
            self._centers = np.array([mu])
            self._centers_idx = np.array([idx_in_X], dtype=int)
            self._coefficients = a2.reshape(1, self.m)
            self.y = y_new  # store y for reuse
            return

        # --- Case 2: incremental update ---
        n = len(self._centers)
        p = n * self.m
        L_old = self.L

        # build B_col
        B_col = self._stack_block_column(self._centers, mu)

        K_nn = self.kernel(mu, mu)
        W = solve_triangular(L_old, B_col, lower=True)

        # Schur complement and its Cholesky decomposition (small, only for the new block)
        S = K_nn - W.T @ W + self.reg * np.eye(self.m)
        L22 = np.linalg.cholesky(S)

        # extend L
        new_p = p + self.m
        L_new = np.zeros((new_p, new_p))
        L_new[:p, :p] = L_old
        L_new[p:new_p, :p] = W.T
        L_new[p:new_p, p:new_p] = L22
        self.L = L_new

        # incremental update of y
        rhs_proj = f_new - W.T @ self.y
        y_new = solve_triangular(L22, rhs_proj, lower=True)
        self.y = np.concatenate([self.y, y_new])  # store for next iteration

        # incremental update of coefficients
        a2 = solve_triangular(L22.T, y_new, lower=False)
        delta_vec = solve_triangular(L_old.T, W @ a2, lower=False)

        a1_flat_old = self._coefficients.reshape(p,)
        a1_flat_new = a1_flat_old - delta_vec
        a_flat_new = np.concatenate([a1_flat_new, a2.reshape(self.m,)])
        self._coefficients = a_flat_new.reshape(n + 1, self.m)

        # update centers and indices
        self._centers = np.vstack([self._centers, mu])
        self._centers_idx = np.concatenate([self._centers_idx, np.array([idx_in_X], dtype=int)])


class VKOGAEstimator(BasicObject):
    """Scikit-learn-style estimator using :class:`~pymor.algorithms.ml.vkoga.VKOGASurrogate`.

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
        self.surrogate_ = None

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
