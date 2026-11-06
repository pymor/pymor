# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Vector fitting."""

import numpy as np
import scipy.linalg as spla
from scipy.optimize import brentq, linear_sum_assignment

from pymor.algorithms.loewner import complete_conjugate_pairs, sample_transfer_function
from pymor.core.base import BasicObject
from pymor.models.iosys import LTIModel


class VectorFittingReductor(BasicObject):
    r"""Vector fitting reductor.

    Computes a rational approximation in pole-residue form

    .. math::
        H_r(s) = \sum_{j = 1}^r \frac{\Phi_j}{s - \lambda_j} + D

    of sampled transfer function data :math:`H(s_i)`, :math:`i = 1, \ldots, \ell`, by
    approximately minimizing the weighted least squares error

    .. math::
        \sum_{i = 1}^{\ell} \rho_i \left\lVert H_r(s_i) - H(s_i) \right\rVert_F^2

    using vector fitting :cite:`GS99`. All entries of the matrix-valued data share the poles
    :math:`\lambda_j`.

    The implementation uses the relaxed non-triviality constraint :cite:`G06` and the numerical
    improvements of :cite:`DGB15`:

    - the residues are eliminated from the linearized least squares problem using a QR
      decomposition shared by all entries of the data, such that only a small least squares
      problem remains for identifying the poles, which is solved using a rank-revealing QR
      decomposition,
    - the residues are computed using a singular value decomposition of the (diagonally scaled)
      Cauchy matrix with high relative accuracy :cite:`D00`, combined with a truncated singular
      value decomposition or Tikhonov regularization,
    - the iteration is stopped based on a backward error bound, and
    - in the MIMO case, the residues are by default restricted to rank one using alternating least
      squares, such that the McMillan degree of :math:`H_r` is :math:`r`.

    Following :cite:`DGB15a`, unstable poles are reflected to the open left half-plane, periodic
    behavior of the poles is detected, and the iterate with the smallest error is returned.
    Using nodes and weights from :func:`h2_quadrature_rule`, the weighted least squares error
    approximates the :math:`\mathcal{H}_2` error.

    Parameters
    ----------
    s
        Sampling points in the complex plane as a 1D |NumPy array|.
    Hs
        Transfer function values at the sampling points `s` as a |NumPy array| of shape
        `(len(s), p, m)` for systems with `p` outputs and `m` inputs or `(len(s),)` for SISO
        systems.
        Use :meth:`generate_samples` to generate data beforehand.
    weights
        Positive weights :math:`\rho_i` in the weighted least squares error as a 1D
        |NumPy array|. If not given, it is set to a vector of ones.
    force_real
        Whether to compute a real model. If `True`, the data is completed with complex conjugates
        (see :func:`~pymor.algorithms.loewner.complete_conjugate_pairs`), poles are real or come
        in complex conjugate pairs, and all computations for identifying the poles are done in
        real arithmetic.

    Attributes
    ----------
    poles_list
        List of poles in each iteration.
    errors
        List of relative weighted least squares errors in each iteration.
    conv_crit
        List of backward error bounds :math:`\theta` (see :cite:`DGB15`, Section 3.1.1) in each
        iteration.
    dists
        List of relative distances of the new poles to the poles in the previous `num_prev`
        iterations.
    poles
        Poles of the returned reduced-order model.
    residues
        Residues of the returned reduced-order model as a |NumPy array| of shape `(r, p, m)`.
    D
        Feedthrough of the returned reduced-order model as a |NumPy array| of shape `(p, m)`.
    """

    generate_samples = staticmethod(sample_transfer_function)

    def __init__(self, s, Hs, weights=None, force_real=True):
        assert isinstance(s, np.ndarray)
        assert s.ndim == 1

        Hs = np.asarray(Hs)
        assert Hs.ndim in (1, 3), 'Hs must contain sample data; use generate_samples to sample a model.'
        assert len(Hs) == len(s)
        if Hs.ndim == 1:
            Hs = Hs[:, np.newaxis, np.newaxis]

        if weights is None:
            weights = np.ones(len(s))
        assert isinstance(weights, np.ndarray)
        assert weights.shape == (len(s),)
        assert np.all(weights > 0)

        # add complex conjugate samples
        if force_real:
            s, Hs, weights = complete_conjugate_pairs(s, Hs, weights)

        self.s = s
        self.Hs = Hs
        self.weights = weights
        self.force_real = force_real
        self.dim_output, self.dim_input = Hs.shape[1:]

    def reduce(self, r=None, initial_poles='log', fit_D=False, residue_rank=1, relax=True,
               force_stability=True, tol=1e-4, maxit=100, num_prev=2, als_maxit=100, noise_level=None):
        r"""Reduce using vector fitting.

        Parameters
        ----------
        r
            Number of poles (if not given, it is `len(initial_poles)`).
        initial_poles
            Initial poles. Can be:

            - `'log'`: weakly damped complex conjugate pairs :math:`-\beta/100 \pm \beta \imath`
              with :math:`\beta` logarithmically spaced over the range of sampled frequencies
              (see :cite:`GS99`, Section 3.2),
            - `'lin'`: as `'log'`, but with linearly spaced :math:`\beta`,
            - `'real'`: logarithmically spaced real poles, which can be preferable for smooth
              data,
            - a 1D |NumPy array| of poles, closed under complex conjugation if
              `force_real` is `True`.

            For odd `r`, a real pole is added to the complex conjugate pairs.
        fit_D
            Whether to fit a constant term :math:`D`. Otherwise, :math:`H_r` is strictly proper.
        residue_rank
            Maximal rank of the residues :math:`\Phi_j`. If `None`, the residues are of full
            rank and the McMillan degree of :math:`H_r` is up to :math:`r \min(p, m)`. Otherwise,
            the residues are computed by truncating the full-rank residues and refining using
            alternating least squares (see :cite:`DGB15`, Section 4.1).
        relax
            Whether to use the relaxed non-triviality constraint :cite:`G06`.
        force_stability
            Whether to reflect unstable poles to the left half-plane in each iteration.
        tol
            Tolerance for the convergence criteria. The iteration stops if the backward error
            bound :math:`\theta` or the relative distance of the new poles to the poles in one
            of the previous `num_prev` iterations is below `tol`. In alternating least squares,
            the iteration stops when the relative error decreases by less than `tol`.
        maxit
            Maximum number of vector fitting iterations.
        num_prev
            Number of previous iterations to compare the new poles to. Values greater than one
            allow to detect periodic behavior.
        als_maxit
            Maximum number of alternating least squares iterations.
        noise_level
            Estimated relative noise level of the data. If given, the residues are computed using
            Tikhonov regularization with the regularization parameter chosen by the Morozov
            discrepancy principle (see :cite:`DGB15`, Section 3.4). Otherwise, a truncated singular
            value decomposition is used.

        Returns
        -------
        rom
            Reduced-order |LTIModel|.
        """
        if isinstance(initial_poles, str):
            assert initial_poles in ('log', 'lin', 'real')
            assert r is not None, 'Either r or initial poles must be given.'
            assert r > 0
            poles = self._initial_poles(r, initial_poles)
        else:
            assert isinstance(initial_poles, np.ndarray)
            assert initial_poles.ndim == 1
            assert r is None or r == len(initial_poles)
            poles = initial_poles.astype(complex)
            if self.force_real:
                assert np.allclose(np.sort_complex(poles), np.sort_complex(poles.conj())), \
                    'initial_poles must be closed under complex conjugation.'
        assert residue_rank is None or residue_rank > 0
        assert tol >= 0
        assert maxit >= 0
        assert num_prev > 0
        assert noise_level is None or noise_level >= 0

        poles = _canonical_poles(poles, self.force_real)
        if force_stability:
            poles = -np.abs(poles.real) + 1j * poles.imag

        self.poles_list = []
        self.errors = []
        self.conv_crit = []
        self.dists = []
        data = self._pole_identification_data()
        with self.logger.block('Starting vector fitting'):
            for it in range(maxit):
                error = self._identify_residues(poles, fit_D, noise_level)[1]
                sigma_residues, poles_new = self._identify_poles(poles, data, fit_D, relax)
                theta = np.sum(np.abs(sigma_residues) / np.abs(poles.real))
                if force_stability:
                    poles_new = -np.abs(poles_new.real) + 1j * poles_new.imag
                dist = min(_pole_distance(poles_new, poles_old)
                           for poles_old in [poles, *self.poles_list[:-num_prev:-1]])
                self.poles_list.append(poles)
                self.errors.append(error)
                self.conv_crit.append(theta)
                self.dists.append(dist)
                self.logger.info(f'Iteration {it + 1}: error={error:.3e}, theta={theta:.3e}, dist={dist:.3e}')
                if theta <= tol or dist <= tol:
                    break
                poles = poles_new
            else:
                if maxit > 0:
                    self.logger.warning(f'Vector fitting did not converge in {maxit} iterations.')

        if self.errors:
            best = int(np.argmin(self.errors))
            poles = self.poles_list[best]
            self.logger.info(f'Using poles from iteration {best + 1}')
        residues = self._identify_residues(poles, fit_D, noise_level)[0]

        p, m = self.dim_output, self.dim_input
        if residue_rank is not None and residue_rank < min(p, m):
            c, b, D = self._als(poles, residues, residue_rank, fit_D, tol, als_maxit)
        else:
            D = residues[-1] if fit_D else np.zeros((p, m))
            c, b = _residue_factors(poles, residues[:len(poles)], self.force_real)
        if self.force_real:
            D = D.real

        self.poles = poles
        self.residues = np.einsum('jpk,jmk->jpm', c, b)
        self.D = D
        error = _relative_error(self.s, self.weights, self.Hs, poles, self.residues, D)
        self.logger.info(f'Relative error of the reduced-order model: {error:.3e}')

        return _pole_residue_to_lti(poles, c, b, D if fit_D else None, self.force_real)

    def _initial_poles(self, r, spacing):
        freqs = np.abs(self.s.imag[self.s.imag != 0])
        if len(freqs) == 0:
            freqs = np.abs(self.s[self.s != 0])
            spacing = 'real'
        w_min, w_max = freqs.min(), freqs.max()
        if spacing == 'real':
            return -np.geomspace(w_min, w_max, r).astype(complex)
        beta = np.geomspace(w_min, w_max, r // 2) if spacing == 'log' else np.linspace(w_min, w_max, r // 2)
        poles = np.concatenate((-beta / 100 + 1j * beta, -beta / 100 - 1j * beta))
        if r % 2 == 1:
            poles = np.append(poles, -np.sqrt(w_min * w_max))
        return poles

    def _pole_identification_data(self):
        """Nodes, square roots of weights and data used for identifying the poles.

        In the real case, conjugate samples are merged such that only nodes in the closed upper
        half-plane remain. For real coefficient vectors, the least squares objective does not
        change except for a constant term.
        """
        s, Hs, weights = self.s, self.Hs, self.weights
        if self.force_real:
            lower = s.imag < 0
            s, idx = np.unique(np.where(lower, s.conj(), s), return_inverse=True)
            Hs_merged = np.zeros((len(s), *Hs.shape[1:]), dtype=complex)
            np.add.at(Hs_merged, idx, weights[:, np.newaxis, np.newaxis]
                      * np.where(lower[:, np.newaxis, np.newaxis], Hs.conj(), Hs))
            weights = np.bincount(idx, weights=weights)
            Hs = Hs_merged / weights[:, np.newaxis, np.newaxis]
        return s, np.sqrt(weights), Hs

    def _identify_poles(self, poles, data, fit_D, relax):
        """Solve the linearized least squares problem and relocate the poles.

        Returns the residues of the scaling function `sigma`, normalized such that its constant
        term is one, and the zeros of `sigma`.
        """
        s, w, Hs = data
        r = len(poles)
        Phi = _basis(s, poles, self.force_real)
        Psi = np.hstack((Phi, np.ones((len(s), 1))))
        if self.force_real:
            nonreal = s.imag > 0

            def stack(M):
                return np.concatenate((M.real, M[nonreal].imag))
        else:
            def stack(M):
                return M

        # eliminate the residues using a single QR decomposition (see [DGB15], Section 3.5.1)
        Q1 = spla.qr(stack(w[:, np.newaxis] * (Psi if fit_D else Phi)), mode='economic')[0]
        blocks = []
        for Hs_ij in Hs.reshape(len(s), -1).T:
            B2 = stack(-(w * Hs_ij)[:, np.newaxis] * Psi)
            B2 -= Q1 @ (Q1.conj().T @ B2)
            blocks.append(spla.qr(B2, mode='r')[0][:r + 1])
        S = np.vstack(blocks)

        # relaxed non-triviality constraint: weighted mean of (the real part of) sigma is one
        d = 0
        if relax:
            mean = w**2 @ Psi / np.sum(w**2)
            if self.force_real:
                mean = mean.real
            scale = spla.norm(w[:, np.newaxis, np.newaxis] * Hs)
            x = _lstsq(np.vstack((S, scale * mean)), np.concatenate((np.zeros(len(S)), [scale])))
            c, d = x[:-1], x[-1]
        if abs(d) < 1e-8:
            if relax:
                self.logger.info('Relaxed constraint failed, using sigma(infinity) = 1.')
            c, d = _lstsq(S[:, :-1], -S[:, -1]), 1
        c = c / d

        # zeros of sigma
        if self.force_real:
            A, b = _real_diag(poles)
            zeros = spla.eigvals(A - np.outer(b, c))
            c = _complex_coefficients(c, poles)
        else:
            zeros = spla.eigvals(np.diag(poles) - np.outer(np.ones(r), c))

        return c, _canonical_poles(zeros, self.force_real)

    def _identify_residues(self, poles, fit_D, noise_level):
        """Solve the least squares problem for the residues (see [DGB15], Section 3.4).

        The least squares problem is solved using the SVD of the Cauchy matrix with high relative
        accuracy, followed by one step of iterative refinement with the residual computed using the
        Cauchy matrix itself. The refinement step recovers the small residual of a column-wise
        backward stable solver for graded matrices.

        Returns the full-rank residues (and `D` if `fit_D` is `True`) as a |NumPy array| of shape
        `(len(poles) + fit_D, p, m)` and the relative error.
        """
        w = np.sqrt(self.weights)
        G = _cauchy_matrix(self.s, poles, w, fit_D)
        U, sv, V = _cauchy_svd(self.s, poles, w, fit_D)
        H = (w[:, np.newaxis, np.newaxis] * self.Hs).reshape(len(self.s), -1)
        filt, damp = _filter_factors(sv, U.conj().T @ H, spla.norm(H), noise_level)
        X = V @ (filt[:, np.newaxis] * (U.conj().T @ H))
        X += V @ (filt[:, np.newaxis] * (U.conj().T @ (H - G @ X)) - damp[:, np.newaxis] * (V.conj().T @ X))
        X = X.reshape(len(X), *self.Hs.shape[1:])
        if self.force_real:
            X[:len(poles)] = _symmetrize(X[:len(poles)], poles)
            if fit_D:
                X[-1] = X[-1].real
        D = X[-1] if fit_D else 0
        error = _relative_error(self.s, self.weights, self.Hs, poles, X[:len(poles)], D)
        return X, error

    def _als(self, poles, X, k, fit_D, tol, maxit):
        """Alternating least squares for residues of rank `k` (see [DGB15], Section 4.1).

        Starts from the truncated singular value decompositions of the full-rank residues `X`.
        The least squares problems are reduced using a QR decomposition of the (column-scaled)
        Cauchy matrix.
        """
        r = len(poles)
        p, m = self.dim_output, self.dim_input
        c, b = _residue_factors(poles, X[:r], self.force_real, k)
        D = X[-1] if fit_D else np.zeros((p, m))

        w = np.sqrt(self.weights)
        G = _cauchy_matrix(self.s, poles, w, fit_D)
        norms = spla.norm(G, axis=0)
        Q, R = spla.qr(G / norms, mode='economic')
        R *= norms
        H = w[:, np.newaxis, np.newaxis] * self.Hs
        QhH = np.tensordot(Q.conj().T, H, 1)
        q = len(R)
        norm_H = spla.norm(H)
        res_perp = max(norm_H**2 - spla.norm(QhH)**2, 0)

        def error():
            Z = np.einsum('jpk,jmk->jpm', c, b)
            if fit_D:
                Z = np.concatenate((Z, D[np.newaxis]))
            return np.sqrt(spla.norm(np.tensordot(R, Z, 1) - QhH)**2 + res_perp) / norm_H

        def solve(factors, rhs):
            # least squares problem for the other factors (and D) with `factors` fixed
            M = np.vstack([(R[:, :r, np.newaxis] * f).reshape(q, -1) for f in factors])
            if fit_D:
                M = np.hstack((M, np.kron(np.eye(len(factors)), R[:, r:])))
            sol = spla.lstsq(M, rhs)[0]
            new = sol[:r * k].reshape(r, k, -1).transpose(0, 2, 1)
            if self.force_real:
                new = _symmetrize(new, poles)
            return new, sol[r * k:]

        err = error()
        self.logger.info(f'ALS iteration 0: error={err:.3e}')
        for it in range(maxit):
            c, D_T = solve(b.transpose(1, 0, 2), QhH.transpose(2, 0, 1).reshape(m * q, p))
            if fit_D:
                D = D_T.T
            b, D_new = solve(c.transpose(1, 0, 2), QhH.transpose(1, 0, 2).reshape(p * q, m))
            if fit_D:
                D = D_new.real if self.force_real else D_new
            err_old, err = err, error()
            self.logger.info(f'ALS iteration {it + 1}: error={err:.3e}')
            if err_old - err <= tol * err:
                break
        return c, b, D


def h2_quadrature_rule(num_nodes, scale=1.):
    r"""Quadrature rule for the :math:`\mathcal{H}_2` inner product.

    Returns nodes and weights

    .. math::
        \xi_j = \imath L \cot\left(\frac{j \pi}{n + 1}\right), \quad
        w_j = \frac{L}{2 (n + 1) \sin^2\left(\frac{j \pi}{n + 1}\right)}, \quad
        j = 1, \ldots, n,

    such that

    .. math::
        \sum_{j = 1}^n w_j \left\lVert F(\xi_j) \right\rVert_F^2
        \approx \frac{1}{2 \pi} \int_{-\infty}^{\infty}
        \left\lVert F(\imath \omega) \right\rVert_F^2 \operatorname{d}\!\omega
        = \lVert F \rVert_{\mathcal{H}_2}^2,

    obtained using the transformation :math:`\omega = L \cot(t)` (see :cite:`DGB15a`,
    Section 3.4). The terms at :math:`t = 0` and :math:`t = \pi` are omitted.
    Using these nodes and weights in |VectorFittingReductor| gives a discretized
    :math:`\mathcal{H}_2` approximation problem.

    Parameters
    ----------
    num_nodes
        Number of nodes :math:`n`.
    scale
        Positive scaling parameter :math:`L`, which should be adapted to the frequency range of
        interest. Half of the nodes have magnitude less than `scale`.

    Returns
    -------
    nodes
        1D |NumPy array| of purely imaginary nodes, closed under complex conjugation.
    weights
        1D |NumPy array| of positive weights.
    """
    assert num_nodes > 0
    assert scale > 0
    t = np.arange(1, num_nodes + 1) * np.pi / (num_nodes + 1)
    omega = scale / np.tan(t)
    weights = scale / (2 * (num_nodes + 1) * np.sin(t)**2)
    # enforce exact symmetry
    omega = (omega - omega[::-1]) / 2
    weights = (weights + weights[::-1]) / 2
    return 1j * omega, weights


def _canonical_poles(poles, real):
    """Sort poles.

    In the real case, real poles come first, followed by complex conjugate pairs, each with the
    pole in the upper half-plane first.
    """
    if not real:
        return poles[np.lexsort((poles.real, poles.imag))]
    real_poles = np.sort(poles[poles.imag == 0].real)
    upper = poles[poles.imag > 0]
    upper = upper[np.lexsort((upper.real, upper.imag))]
    assert len(upper) == np.sum(poles.imag < 0)
    poles = np.empty(len(real_poles) + 2 * len(upper), dtype=complex)
    poles[:len(real_poles)] = real_poles
    poles[len(real_poles)::2] = upper
    poles[len(real_poles) + 1::2] = upper.conj()
    return poles


def _basis(s, poles, real):
    """Partial fractions evaluated at `s`.

    In the real case, a complex conjugate pair of partial fractions is replaced by their sum and
    `1j` times their difference (see [GS99], Appendix A).
    """
    C = 1 / (s[:, np.newaxis] - poles)
    if real:
        upper = np.flatnonzero(poles.imag > 0)
        C[:, upper], C[:, upper + 1] = C[:, upper] + C[:, upper + 1], 1j * (C[:, upper] - C[:, upper + 1])
    return C


def _complex_coefficients(c, poles):
    """Convert coefficients with respect to the real basis to residues."""
    c = c.astype(complex)
    upper = np.flatnonzero(poles.imag > 0)
    c[upper], c[upper + 1] = c[upper] + 1j * c[upper + 1], c[upper] - 1j * c[upper + 1]
    return c


def _real_diag(poles):
    """Real block-diagonal matrix with eigenvalues `poles` and the corresponding input vector.

    For a complex conjugate pair `a + 1j * b` and `a - 1j * b`, the diagonal block is
    `[[a, b], [-b, a]]` and the entries of the input vector are `(2, 0)` (see [GS99], Appendix B).
    """
    A = np.diag(poles.real)
    b = np.ones(len(poles))
    upper = np.flatnonzero(poles.imag > 0)
    A[upper, upper + 1] = poles[upper].imag
    A[upper + 1, upper] = -poles[upper].imag
    b[upper] = 2
    b[upper + 1] = 0
    return A, b


def _symmetrize(X, poles):
    """Make residues of complex conjugate poles complex conjugate and of real poles real."""
    X = X.copy()
    upper = np.flatnonzero(poles.imag > 0)
    X[upper] = (X[upper] + X[upper + 1].conj()) / 2
    X[upper + 1] = X[upper].conj()
    X[poles.imag == 0] = X[poles.imag == 0].real
    return X


def _lstsq(M, rhs):
    """Basic least squares solution using a rank-revealing QR decomposition.

    The columns of `M` are scaled to unit norm. Unknowns corresponding to truncated columns are
    set to zero, such that, in pole identification, the corresponding poles do not move.
    """
    norms = spla.norm(M, axis=0)
    norms[norms == 0] = 1
    M = M / norms
    rcond = max(M.shape) * np.finfo(float).eps
    Q, R, piv = spla.qr(M, mode='economic', pivoting=True)
    k = np.count_nonzero(np.abs(np.diag(R)) > rcond * np.abs(R[0, 0]))
    x = np.zeros(M.shape[1], dtype=np.result_type(M, rhs))
    x[piv[:k]] = spla.solve_triangular(R[:k, :k], Q[:, :k].conj().T @ rhs)
    return x / norms


def _pole_distance(poles1, poles2):
    """Maximal relative distance between optimally matched poles."""
    dist = np.abs(poles1[:, np.newaxis] - poles2) / np.maximum(np.abs(poles1), np.finfo(float).tiny)[:, np.newaxis]
    rows, cols = linear_sum_assignment(dist)
    return dist[rows, cols].max()


def _cauchy_matrix(s, poles, w, constant):
    """Diagonally scaled Cauchy matrix, with the column `w` appended if `constant` is `True`."""
    G = w[:, np.newaxis] / (s[:, np.newaxis] - poles)
    return np.hstack((G, w[:, np.newaxis])) if constant else G


def _relative_error(s, weights, Hs, poles, residues, D):
    Hr = np.tensordot(1 / (s[:, np.newaxis] - poles), residues, 1) + D
    w = np.sqrt(weights)[:, np.newaxis, np.newaxis]
    return spla.norm(w * (Hr - Hs)) / spla.norm(w * Hs)


def _cauchy_ldu(x, y, w, constant):
    """LDU decomposition of a diagonally scaled Cauchy matrix with complete pivoting.

    Computes `L`, `d`, `U`, `rows`, `cols` such that `G[rows][:, cols] = L @ np.diag(d) @ U`,
    where `G = w[:, np.newaxis] / (x[:, np.newaxis] - y)`, with a column `w` appended if
    `constant` is `True`. The Schur complements are computed using the Cauchy structure such that
    all entries of `L`, `d`, and `U` are computed to high relative accuracy (see [D00],
    Algorithm 3).
    """
    G = _cauchy_matrix(x, y, w, constant)
    x = x.copy()
    y = np.append(y, 0) if constant else y.copy()
    inf = np.arange(len(y)) >= len(y) - constant
    m, n = G.shape
    rows = np.arange(m)
    cols = np.arange(n)
    q = min(m, n)
    for k in range(q):
        i, j = np.unravel_index(np.argmax(np.abs(G[k:, k:])), (m - k, n - k))
        i, j = i + k, j + k
        for v in (G, x, rows):
            v[[k, i]] = v[[i, k]]
        G[:, [k, j]] = G[:, [j, k]]
        for v in (y, inf, cols):
            v[[k, j]] = v[[j, k]]
        if G[k, k] == 0:
            q = k
            break
        G[k + 1:, k] /= G[k, k]
        G[k, k + 1:] /= G[k, k]
        if inf[k]:
            G[k + 1:, k + 1:] *= (x[k] - x[k + 1:])[:, np.newaxis] / (x[k] - y[k + 1:])
        else:
            col_factors = np.ones(n - k - 1, dtype=complex)
            fin = ~inf[k + 1:]
            col_factors[fin] = (y[k] - y[k + 1:][fin]) / (x[k] - y[k + 1:][fin])
            G[k + 1:, k + 1:] *= ((x[k + 1:] - x[k]) / (x[k + 1:] - y[k]))[:, np.newaxis] * col_factors
    L = np.tril(G[:, :q], -1) + np.eye(m, q)
    d = np.diag(G)[:q].copy()
    U = np.triu(G[:q], 1) + np.eye(q, n)
    return L, d, U, rows, cols


def _cauchy_svd(x, y, w, constant):
    """Singular value decomposition of a diagonally scaled Cauchy matrix.

    Computes the SVD `G = U @ np.diag(sv) @ V.conj().T` of `G` (see :func:`_cauchy_ldu`) with
    high relative accuracy from its LDU decomposition (see [D00] and [DGESVD99], Algorithm 3.1).
    """
    L, d, U, rows, cols = _cauchy_ldu(x, y, w, constant)
    Q, R, piv = spla.qr(L * d, mode='economic', pivoting=True)
    W = R @ U[piv]
    Z, J = _one_sided_jacobi(W.conj().T)
    sv = spla.norm(Z, axis=0)
    order = np.argsort(sv)[::-1]
    sv = sv[order]
    k = np.count_nonzero(sv)
    sv = sv[:k]
    order = order[:k]
    U_svd = np.empty((len(x), k), dtype=complex)
    U_svd[rows] = Q @ J[:, order]
    V_svd = np.empty((len(cols), k), dtype=complex)
    V_svd[cols] = Z[:, order] / sv
    return U_svd, sv, V_svd


def _one_sided_jacobi(Z, max_sweeps=100):
    """One-sided Jacobi method.

    Computes `Z @ J` with orthogonal columns, where `J` is unitary, using the round-robin
    ordering. For `Z` of the form `Y @ np.diag(d)` with well-conditioned `Y`, the column norms of
    `Z @ J` are the singular values of `Z` with high relative accuracy.
    """
    Z = Z.astype(complex)
    n, q = Z.shape
    J = np.eye(q, dtype=complex)
    tol = np.sqrt(n) * np.finfo(float).eps
    rounds = _round_robin(q)
    for _ in range(max_sweeps):
        rotated = False
        for i, j in rounds:
            a = np.sum(np.abs(Z[:, i])**2, axis=0)
            b = np.sum(np.abs(Z[:, j])**2, axis=0)
            g = np.sum(Z[:, i].conj() * Z[:, j], axis=0)
            abs_g = np.abs(g)
            mask = abs_g > tol * np.sqrt(a * b)
            if not np.any(mask):
                continue
            rotated = True
            i, j, a, b, g, abs_g = i[mask], j[mask], a[mask], b[mask], g[mask], abs_g[mask]
            zeta = (b - a) / (2 * abs_g)
            t = np.where(zeta >= 0, 1, -1) / (np.abs(zeta) + np.hypot(1, zeta))
            cos = 1 / np.sqrt(1 + t**2)
            sin = cos * t
            phase = (g / abs_g).conj()
            for M in (Z, J):
                Mi, Mj = M[:, i], M[:, j] * phase
                M[:, i], M[:, j] = cos * Mi - sin * Mj, sin * Mi + cos * Mj
        if not rotated:
            break
    return Z, J


def _round_robin(n):
    """Round-robin ordering of index pairs for the one-sided Jacobi method."""
    m = n + n % 2
    idx = np.arange(m)
    rounds = []
    for _ in range(m - 1):
        i, j = idx[:m // 2], idx[m // 2:][::-1]
        mask = (i < n) & (j < n)
        rounds.append((i[mask], j[mask]))
        idx = np.concatenate((idx[:1], idx[-1:], idx[1:-1]))
    return rounds


def _filter_factors(sv, UhH, norm_H, noise_level):
    """Filter factors for a regularized least squares solution.

    Returns `filt` and `damp` such that the solution is `V @ (filt * (U^H H))` and its fixed point
    condition for iterative refinement is `filt * (U^H R) - damp * (V^H X) = 0` with residual `R`.
    If `noise_level` is `None`, a truncated SVD is used. Otherwise, the Tikhonov regularization
    parameter `mu` is chosen such that the residual norm is `noise_level * norm_H` (Morozov
    discrepancy principle).
    """
    filt = np.where(sv > len(sv) * np.finfo(float).eps * sv[0], 1 / sv, 0)
    damp = np.zeros_like(sv)
    if noise_level is None:
        return filt, damp
    proj = spla.norm(UhH, axis=1)**2
    res_min = max(norm_H**2 - np.sum(proj), 0)
    target = (noise_level * norm_H)**2
    if target <= res_min:
        return filt, damp
    if target >= norm_H**2:
        return np.zeros_like(sv), np.ones_like(sv)

    def residual(log_mu):
        mu2 = np.exp(2 * log_mu)
        return res_min + np.sum((mu2 / (sv**2 + mu2))**2 * proj) - target

    mu2 = np.exp(2 * brentq(residual, np.log(sv[-1]) - 20, np.log(sv[0]) + 20))
    return sv / (sv**2 + mu2), mu2 / (sv**2 + mu2)


def _residue_factors(poles, residues, real, rank=None):
    """Factorize residues as `residues[j] = c[j] @ b[j].T`.

    If `rank` is `None`, the factors have the numerical rank of the residues (padded with zero
    columns to the maximal rank). Otherwise, the truncated singular value decompositions of rank
    `rank` are used.
    """
    r, p, m = residues.shape
    k = min(p, m) if rank is None else rank
    c = np.zeros((r, p, k), dtype=complex)
    b = np.zeros((r, m, k), dtype=complex)
    for j, (pole, residue) in enumerate(zip(poles, residues, strict=True)):
        if real and pole.imag < 0:
            c[j], b[j] = c[j - 1].conj(), b[j - 1].conj()
            continue
        U, sv, Vh = spla.svd(residue.real if real and pole.imag == 0 else residue, full_matrices=False)
        if rank is None:
            sv = np.where(sv > max(p, m) * np.finfo(float).eps * sv[0], sv, 0)
        c[j] = U[:, :k] * np.sqrt(sv[:k])
        b[j] = Vh[:k].T * np.sqrt(sv[:k])
    if rank is None:
        nonzero = np.any(c != 0, axis=(0, 1))
        c, b = c[:, :, nonzero], b[:, :, nonzero]
    return c, b


def _pole_residue_to_lti(poles, c, b, D, real):
    """Realization of the pole-residue form with residues `c[j] @ b[j].T`.

    Zero columns of the residue factors are removed. In the real case, a complex conjugate pair
    is realized using a real 2-by-2 block (see [GS99], Appendix B).
    """
    A, B, C = [], [], []
    for pole, c_j, b_j in zip(poles, c, b, strict=True):
        nonzero = np.any(c_j != 0, axis=0) | np.any(b_j != 0, axis=0)
        c_j, b_j = c_j[:, nonzero], b_j[:, nonzero]
        I = np.eye(c_j.shape[1])
        if not real:
            A.append(pole * I)
            B.append(b_j.T)
            C.append(c_j)
        elif pole.imag == 0:
            A.append(pole.real * I)
            B.append(b_j.T.real)
            C.append(c_j.real)
        elif pole.imag > 0:
            A.append(np.block([[pole.real * I, pole.imag * I], [-pole.imag * I, pole.real * I]]))
            B.append(np.vstack((2 * b_j.T.real, -2 * b_j.T.imag)))
            C.append(np.hstack((c_j.real, c_j.imag)))
    return LTIModel.from_matrices(spla.block_diag(*A), np.vstack(B), np.hstack(C), D)
