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
        assert criterion in ('fp', 'f', 'p')
        self.__auto_init(locals())
        self.X_train = np.asarray(X_train)
        self.F_train = np.asarray(F_train)
        self.N, self.m = self.F_train.shape

        self._centers = None
        self._centers_idx = None
        self._coefficients = None
        self.L = None  # block-Cholesky factor of the kernel matrix used for the Newton basis

    def _stack_block_column(self, X_centers, x_new):
        """Stack K(c_j, x_new) for j over X_centers in order. Result shape (n*m, m)."""
        if X_centers is None:
            return np.zeros((0, self.m))
        n = len(X_centers)
        B = np.zeros((n * self.m, self.m))
        for j, cj in enumerate(X_centers):
            B[j*self.m:(j+1)*self.m, :] = self.kernel(cj, x_new)
        return B

    def _stack_rhs_from_indices(self, centers_idx):
        """Stack F_train[centers_idx] in same order => (n*m,)."""
        if len(centers_idx) == 0:
            return np.zeros((0,))
        blocks = [self.F_train[idx].reshape(self.m,) for idx in centers_idx]
        return np.concatenate(blocks, axis=0)

    def predict(self, X):
        X = np.asarray(X)
        if self._centers is None:
            return np.zeros((X.shape[0], self.m))
        y = np.zeros((X.shape[0], self.m))
        for j, cj in enumerate(self._centers):
            K_vals = np.array([self.kernel(x, cj) for x in X])  # (N_new, m, m)
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
            v = np.linalg.solve(self.L, B_col)
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
        mu = np.asarray(mu)

        # find matching training index
        matches = np.where(np.all(np.isclose(self.X_train, mu), axis=1))[0]
        if matches.size > 0:
            idx_in_X = int(matches[0])
        else:
            idx_in_X = int(np.argmin(np.linalg.norm(self.X_train - mu, axis=1)))
            self.logger.warn(f'mu={mu} not in X_train; using nearest neighbor index {idx_in_X}')

        # build the new block column and diagonal
        B_col = self._stack_block_column(self._centers, mu)  # shape (n*m, m)
        K_nn = self.kernel(mu, mu)

        # compute W and L22 incrementally
        if self.L is None:
            S = K_nn + self.reg * np.eye(self.m)
            L22 = np.linalg.cholesky(S)
            self.L = L22
            y_new = np.linalg.solve(L22, self.F_train[idx_in_X])
            coefficient_new = np.linalg.solve(L22.T, y_new).reshape(1, -1)
            self._centers = mu[None, :]
            self._centers_idx = np.array([idx_in_X], dtype=int)
            self._coefficients = coefficient_new
            return

        # otherwise: incremental update
        L_old = self.L
        p = L_old.shape[0]
        # solve for W
        W = np.linalg.solve(L_old, B_col)
        # compute Schur complement
        S = K_nn - W.T @ W + self.reg * np.eye(self.m)
        L22 = np.linalg.cholesky(S)

        # update L
        new_p = p + self.m
        new_L = np.zeros((new_p, new_p))
        new_L[:p, :p] = L_old
        new_L[p:new_p, :p] = W.T
        new_L[p:new_p, p:new_p] = L22
        self.L = new_L

        # incremental coefficient update (no recomputation)
        # old system: L_old L_old^T coefficients_flat_old = rhs_old
        # new system:
        # [L_old   0  ][L_old^T  W ][coefficients_old]   = [rhs_old]
        # [ W^T  L22 ][W^T  L22^T][coefficient_new]     [f_new]

        rhs_old = self._stack_rhs_from_indices(self._centers_idx)
        f_new = self.F_train[idx_in_X].reshape(self.m,)

        # solve for y_old = L_old^{-1} rhs_old
        y_old = np.linalg.solve(L_old, rhs_old)
        # compute the new projected residual
        rhs_proj = f_new - W.T @ y_old
        y_new = np.linalg.solve(L22, rhs_proj)

        # now update y_full and back-substitute to get _coefficients
        y_full = np.concatenate([y_old, y_new])
        coefficients_flat = np.linalg.solve(self.L.T, y_full)
        self._coefficients = coefficients_flat.reshape(-1, self.m)

        # append the new center
        self._centers = np.vstack([self._centers, mu])
        self._centers_idx = np.concatenate([self._centers_idx, [idx_in_X]])


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
