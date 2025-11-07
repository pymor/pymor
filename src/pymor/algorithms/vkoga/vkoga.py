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
        assert criterion in ('fp', 'f', 'p', 'f/p')
        self.__auto_init(locals())
        self.X_train = np.asarray(X_train)
        self.F_train = np.asarray(F_train)
        if self.F_train.ndim == 1:
            self.F_train = self.F_train.reshape((-1, 1))
        self.N, self.m = self.F_train.shape

        self._centers = None
        self._centers_idx = None
        self._coefficients = None
        self._L = None  # block-Cholesky factor of the kernel matrix
        self._init_power_function_evals()

    def predict(self, X):
        X = np.asarray(X)
        if self._centers is None:
            return np.zeros((X.shape[0], self.m))
        y = np.zeros((X.shape[0], self.m))
        for j, cj in enumerate(self._centers):
            K_vals = np.array([self.kernel(x, cj) for x in X])
            y += np.einsum('nij,j->ni', K_vals, self._coefficients[j])
        return y

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

        if self.criterion in ('fp', 'f/p', 'f'):
            Ypred = self.predict(Xc)
            Fres = self.F_train[idxs] - Ypred
            res_norms = np.linalg.norm(Fres, axis=1)
            if self.criterion == 'fp':
                scores = res_norms * np.sqrt(self._power2)
            elif self.criterion == 'f':
                scores = res_norms
            elif self.criterion == 'f/p':
                scores = np.zeros_like(res_norms)
                nonzero_power = np.nonzero(self._power2)
                scores[nonzero_power] = res_norms[nonzero_power] / np.sqrt(self._power2[nonzero_power])
        elif self.criterion == 'p':
            scores = np.sqrt(self._power2)

        if return_all_values:
            return scores
        idx = int(np.argmax(scores))
        return float(scores[idx]), mus[idx]

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

        using a numerically stable Newton basis expansion.

        In the following derivation we leave out the regularization term for simplicity.
        If :math:`K_n` denotes the full block kernel matrix, we can write :math:`K_{n+1}` as

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
                W^\top & L_{22}
            \end{bmatrix},

        where :math:`W=L_n^{-1}B` and :math:`L_{22}` is the Cholesky decomposition
        of :math:`k_{n+1,n+1}-W^\top W`.

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
                B^\top a_1+k_{n+1,n+1}a_2 &= f_2.
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

        Regarding the power function (measuring for :math:`x\in\mathbb{R}^d` how well
        the current subspace can represent :math:`k(\,\cdot\,,x)`, i.e. the projection
        error in the reproducing kernel Hilbert space):
        The power function in matrix form is defined as

        .. math::
            \begin{align*}
                P_n^2(x) = k(x,x) - k(x,X_n)k(X_n,X_n)^{-1}k(X_n,x).
            \end{align*}

        Since we would like to use the power function as selection criterion for centers
        within the greedy iteration, we consider a scalar version by taking the trace:

        .. math::
            \begin{align*}
                p_n^2(x) = \operatorname{trace}(P_n^2(x)).
            \end{align*}

        We are going to track this quantity evaluated at all training points during the
        iterations of the greedy algorithm. In order to do so, we also maintain an array
        storing the Newton basis evaluations of the current basis at the training points:

        .. math::
            \begin{align*}
                V_{n,i} = L_n^{-1}k(X_n,y_i)
            \end{align*}

        for all training points :math:`y_i`.

        Given the first center :math:`x_1`, we initialize the (squared) power function
        values as

        .. math::
            \begin{align*}
                p_n^2(y_i) = \operatorname{trace}(k(y_i,y_i)-V_{1,i}^\top V_{1,i}),
            \end{align*}

        where :math:`V_{1,i} = L_{22}^{-1}k(x_1,y_i)`.

        The incremental updates of :math:`V_{n,i}` to :math:`V_{n+1,i}` and the power
        function values is then performed in the following way:
        Define :math:`\Delta_i=k(x_{n+1},y_i)-W^\top V_{n,i}`
        and :math:`\Xi=L_{22}^{-1}\Delta_i`.
        Then, we have

        .. math::
            \begin{align*}
                p_{n+1}^2(y_i) = p_n^2(y_i) - \lVert \Xi_i\rVert_F^2
            \end{align*}

        and

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

        f_new = self.F_train[idx_in_X].reshape(self.m,)

        # --- Case 1: first center ---
        if self._L is None or self._centers is None:
            K_nn = self.kernel(mu, mu) + self.reg * np.eye(self.m)
            L22 = np.linalg.cholesky(K_nn)
            y_new = solve_triangular(L22, f_new, lower=True)
            a2 = solve_triangular(L22.T, y_new, lower=False)
            self._L = L22
            self._centers = np.array([mu])
            self._centers_idx = np.array([idx_in_X], dtype=int)
            self._coefficients = a2.reshape(1, self.m)
            self.y = y_new  # store y for reuse
            self._init_power_function_evals_after_first_center(mu, L22)
            return

        # --- Case 2: incremental update ---
        n = len(self._centers)
        p = n * self.m
        L_old = self._L

        # build B_col
        B_col = np.zeros((n * self.m, self.m))
        for j, cj in enumerate(self._centers):
            B_col[j*self.m:(j+1)*self.m, :] = self.kernel(cj, mu)

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
        self._L = L_new

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

        # update power function evaluations on training set
        self._update_power_function_evals(mu, W, L22)

        # update centers and indices
        self._centers = np.vstack([self._centers, mu])
        self._centers_idx = np.concatenate([self._centers_idx, np.array([idx_in_X], dtype=int)])

    def _init_power_function_evals(self):
        """Initialize power2 and V when no centers were selected yet."""
        # power2 = trace(k(x,x)) for each training point
        self._power2 = np.empty(len(self.X_train))
        for i, x in enumerate(self.X_train):
            self._power2[i] = np.trace(self.kernel(x, x)) + self.m * self.reg
        # no V yet, we will create V after the first center is added
        self._V = None

    def _init_power_function_evals_after_first_center(self, mu, L22):
        """After adding the first center, compute V and power2."""
        N = len(self.X_train)
        m = self.m
        K_mu_X = np.array([self.kernel(mu, x) for x in self.X_train])
        # V_i = L22^{-1} k(mu, y_i) for each i.
        rhs = K_mu_X.transpose(1, 0, 2).reshape(m, N * m)
        sol = solve_triangular(L22, rhs, lower=True)
        V_new = sol.reshape(m, N, m).transpose(1, 0, 2)

        # compute power2[i]: trace(K_xx - V_i.T @ V_i)
        power2 = np.empty(N)
        for i, x in enumerate(self.X_train):
            S = self.kernel(x, x) - V_new[i].T @ V_new[i]
            power2[i] = np.trace(S) + self.reg * m

        self._V = V_new.reshape(N, m, m)
        self._power2 = np.maximum(power2, 0.0)

    def _update_power_function_evals(self, mu, W, L22):
        """Incrementally update self._power2 and self._V when adding center mu."""
        N = len(self.X_train)
        m = self.m
        V_old = self._V
        p_old = V_old.shape[1]
        # compute k(mu, y_i) for all training points y_i
        K_mu_X = np.array([self.kernel(mu, x) for x in self.X_train])

        # Delta_i = k(mu,y_i) - M_i
        Delta = K_mu_X - np.einsum('ap,npj->naj', W.T, V_old)

        # for each i, compute Xi_i = solve(L22, Delta_i)
        rhs = Delta.transpose(1, 0, 2).reshape(m, N * m)
        sol = solve_triangular(L22, rhs, lower=True, check_finite=False)
        Xi = sol.reshape(m, N, m).transpose(1, 0, 2)

        # reduction: norms = ||Xi||_F^2 = sum over entries^2
        norms = np.sum(Xi * Xi, axis=(1, 2))

        # update power2: p_{n+1}^2(x) = p_n^2(x) - norms
        self._power2 = np.maximum(self._power2 - norms, 0.0)

        # update V: append Xi as the last block of rows
        p_new = p_old + m
        V_new = np.zeros((N, p_new, m))
        V_new[:, :p_old, :] = V_old
        V_new[:, p_old:p_old + m, :] = Xi
        self._V = V_new


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
