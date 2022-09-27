# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.cache import CacheableObject, cached
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator, InverseOperator
from pymor.operators.interface import Operator
from pymor.tools.deprecated import Deprecated
from pymor.tools.random import get_seed_seq, new_rng


class RandomizedRangeFinder(CacheableObject):
    def __init__(self, A, subspace_iterations=0, range_product=None, source_product=None, lambda_min=None,
                 complex=False):
        assert isinstance(A, Operator)
        if range_product is None:
            range_product = IdentityOperator(A.range)
        else:
            assert isinstance(range_product, Operator)

        if source_product is None:
            source_product = IdentityOperator(A.source)
        else:
            assert isinstance(source_product, Operator)

        assert 0 <= subspace_iterations and isinstance(subspace_iterations, int)
        assert isinstance(complex, bool)

        self.__auto_init(locals())
        self._l = 0
        self._Q = [self.A.range.empty()]
        for _ in range(subspace_iterations):
            self._Q.append(self.A.source.empty())
            self._Q.append(self.A.range.empty())
        self._Q = tuple(self._Q)
        self.testvecs = self.A.source.empty()
        self._basis_rng_real = new_rng(get_seed_seq().spawn(1)[0])
        self._test_rng_real = new_rng(get_seed_seq().spawn(1)[0])
        if complex:
            self._basis_rng_imag = new_rng(get_seed_seq().spawn(1)[0])
            self._test_rng_imag = new_rng(get_seed_seq().spawn(1)[0])

    @cached
    def _lambda_min(self):
        if isinstance(self.source_product, IdentityOperator):
            return 1
        elif self.lambda_min is None:
            with self.logger.block('Estimating minimum singular value of source_product...'):
                def mv(v):
                    return self.source_product.apply(self.source_product.source.from_numpy(v)).to_numpy()

                def mvinv(v):
                    return self.source_product.apply_inverse(self.source_product.range.from_numpy(v)).to_numpy()
                L = LinearOperator((self.source_product.source.dim, self.source_product.range.dim), matvec=mv)
                Linv = LinearOperator((self.source_product.range.dim, self.source_product.source.dim), matvec=mvinv)
                return eigsh(L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv)[0]
        else:
            return self.lambda_min

    def _draw_test_vector(self, n):
        with self._test_rng_real:
            W = self.A.source.random(n, distribution='normal')
            if self.complex:
                with self._test_rng_imag:
                    W += 1j * self.A.source.random(n, distribution='normal')
        self.testvecs.append(self.A.apply(W))

    def _maxnorm(self, basis_size, num_testvecs):
        if len(self.testvecs) < num_testvecs:
            self._draw_test_vector(num_testvecs - len(self.testvecs))

        W, Q = self.testvecs[:num_testvecs].copy(), self._find_range(basis_size=basis_size, tol=None)
        W -= Q.lincomb(Q.inner(W, self.range_product).T)
        return np.max(W.norm(self.range_product))

    @cached
    def _c_est(self, num_testvecs, p_fail):
        c = np.sqrt(2 * self._lambda_min()) \
            * erfinv((p_fail / min(self.A.source.dim, self.A.range.dim)) ** (1 / num_testvecs))
        return 1 / c

    def estimate_error(self, basis_size, num_testvecs=20, p_fail=1e-14):
        assert isinstance(basis_size, int) and basis_size > 0
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Requested basis is larger than the rank of the operator!')
            self.logger.info('Proceeding with maximum operator rank...')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert 0 < num_testvecs and isinstance(num_testvecs, int)
        assert 0 < p_fail

        err = self._c_est(num_testvecs, p_fail) * self._maxnorm(basis_size, num_testvecs)
        self.logger.info(f'estimated error: {err:.10f}')

        return err

    def _extend_basis(self, n=1):
        self.logger.info(f'Appending {n} basis vector{"s" if n > 1 else ""}.')

        with self._basis_rng_real:
            W = self.A.source.random(n, distribution='normal')
        if self.complex:
            with self._basis_rng_imag:
                W += 1j * self.A.source.random(n, distribution='normal')

        self._Q[0].append(self.A.apply(W))
        gram_schmidt(self._Q[0], self.range_product, offset=self._l, copy=False)

        for i in range(self.subspace_iterations):
            i = 2*i + 1
            self._Q[i].append(self.source_product.apply_inverse(
                (self.A.apply_adjoint(self.range_product.apply(self._Q[i-1][-n:])))))
            gram_schmidt(self._Q[i], self.source_product, offset=self._l, copy=False)
            self._Q[i+1].append(self.A.apply(self._Q[i][-n:]))
            gram_schmidt(self._Q[i+1], self.range_product, offset=self._l, copy=False)

        self._l += n

    def _find_range(self, basis_size=8, tol=None, num_testvecs=20, p_fail=1e-14, block_size=8, increase_block=True,
                    max_basis_size=500):
        if basis_size > self._l:
            self._extend_basis(basis_size - self._l)

        if tol is not None and self.estimate_error(basis_size, num_testvecs, p_fail) > tol:
            with self.logger.block('Extending range basis adaptively...'):
                max_iter = min(max_basis_size, self.A.source.dim, self.A.range.dim)
                while self._l < max_iter:
                    if increase_block:
                        low = basis_size
                        basis_size += block_size
                        block_size *= 2
                    else:
                        basis_size += block_size
                    basis_size = min(basis_size, max_iter)
                    if self.estimate_error(basis_size, num_testvecs, p_fail) <= tol:
                        break
            if increase_block:
                with self.logger.block('Contracting range basis...'):
                    high = basis_size
                    while True:
                        mid = high - (high - low) // 2
                        if basis_size == mid:
                            break
                        basis_size = mid
                        err = self.estimate_error(basis_size, num_testvecs, p_fail)
                        if err <= tol:
                            high = mid
                        else:
                            low = mid

        return self._Q[-1][:basis_size]

    def find_range(self, basis_size=8, tol=None, num_testvecs=20, p_fail=1e-14, block_size=8, increase_block=True,
                   max_basis_size=500):
        assert isinstance(basis_size, int) and basis_size > 0
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Requested basis is larger than the rank of the operator!')
            self.logger.info('Proceeding with maximum operator rank...')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert tol is None or tol > 0
        assert isinstance(num_testvecs, int) and num_testvecs > 0
        assert p_fail > 0
        assert isinstance(block_size, int) and block_size > 0
        assert isinstance(increase_block, bool)
        assert isinstance(max_basis_size, int) and max_basis_size > 0

        with self.logger.block('Finding range...'):
            Q = self._find_range(basis_size=basis_size, tol=tol, num_testvecs=num_testvecs, p_fail=p_fail,
                                 block_size=block_size, increase_block=increase_block, max_basis_size=max_basis_size)
            self.logger.info(f'Found range of dimension {len(Q)}.')
        return Q


@defaults('tol', 'failure_tolerance', 'num_testvecs')
@Deprecated('RandomizedRangeFinder')
def adaptive_rrf(A, source_product=None, range_product=None, tol=1e-4,
                 failure_tolerance=1e-15, num_testvecs=20, lambda_min=None, iscomplex=False):
    r"""Adaptive randomized range approximation of `A`.

    This is an implementation of Algorithm 1 in :cite:`BS18`.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the norm denotes the
    operator norm. The inner product of the range of `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    B
        |VectorArray| which contains the basis, whose span approximates the range of A.
    """
    RRF = RandomizedRangeFinder(A, subspace_iterations=0, source_product=source_product, range_product=range_product,
                                lambda_min=lambda_min, complex=iscomplex)
    return RRF.find_range(basis_size=1, tol=tol, num_testvecs=num_testvecs, p_fail=failure_tolerance)


@defaults('q', 'l')
@Deprecated('RandomizedRangeFinder')
def rrf(A, source_product=None, range_product=None, q=2, l=8, return_rand=False, iscomplex=False):
    r"""Randomized range approximation of `A`.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `Q` whose vectors form an approximate orthonormal basis for the range of `A`.

    This method is based on algorithm 2 in :cite:`SHB21`.

    Parameters
    ----------
    A
        The |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    q
        The number of power iterations.
    l
        The block size of the normalized power iterations.
    return_rand
        If `True`, the randomly sampled |VectorArray| R is returned.
    iscomplex
        If `True`, the random vectors are chosen complex.

    Returns
    -------
    Q
        |VectorArray| which contains the basis, whose span approximates the range of A.
    R
        The randomly sampled |VectorArray| (if `return_rand` is `True`).
    """
    RRF = RandomizedRangeFinder(A, subspace_iterations=q, source_product=source_product, range_product=range_product,
                                complex=iscomplex)
    Q = RRF.find_range(basis_size=l, tol=None)
    if return_rand:
        return Q, RRF.testvecs
    else:
        return Q


@defaults('p', 'q', 'modes')
def random_generalized_svd(A, range_product=None, source_product=None, modes=6, p=20, q=2):
    r"""Randomized SVD of an |Operator|.

    Viewing `A` as an :math:`m` by :math:`n` matrix, the return value
    of this method is the randomized generalized singular value decomposition of `A`:

    .. math::

        A = U \Sigma V^{-1},

    where the inner product on the range :math:`\mathbb{R}^m` is given by

    .. math::

        (x, y)_S = x^TSy

    and the inner product on the source :math:`\mathbb{R}^n` is given by

    .. math::

        (x, y) = x^TTy.

    This method is based on :cite:`SHB21`.

    Parameters
    ----------
    A
        The |Operator| for which the randomized SVD is to be computed.
    range_product
        Range product |Operator| :math:`S` w.r.t which the randomized SVD is computed.
    source_product
        Source product |Operator| :math:`T` w.r.t which the randomized SVD is computed.
    modes
        The first `modes` approximated singular values and vectors are returned.
    p
        If not `0`, adds `p` columns to the randomly sampled matrix (oversampling parameter).
    q
        If not `0`, performs `q` so-called power iterations to increase the relative weight
        of the first singular values.

    Returns
    -------
    U
        |VectorArray| of approximated left singular vectors.
    s
        One-dimensional |NumPy array| of the approximated singular values.
    Vh
        |VectorArray| of the approximated right singular vectors.
    """
    logger = getLogger('pymor.algorithms.rand_la')

    assert isinstance(A, Operator)

    assert 0 <= modes <= max(A.source.dim, A.range.dim) and isinstance(modes, int)
    assert 0 <= p <= max(A.source.dim, A.range.dim) - modes and isinstance(p, int)
    assert q >= 0 and isinstance(q, int)

    if range_product is None:
        range_product = IdentityOperator(A.range)
    else:
        assert isinstance(range_product, Operator)
        assert range_product.source == range_product.range == A.range

    if source_product is None:
        source_product = IdentityOperator(A.source)
    else:
        assert isinstance(source_product, Operator)
        assert source_product.source == source_product.range == A.source

    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()

    RRF = RandomizedRangeFinder(A, subspace_iterations=q, source_product=source_product, range_product=range_product)
    Q = RRF.find_range(basis_size=modes+p)

    B = A.apply_adjoint(range_product.apply(Q))
    Q_B, R_B = gram_schmidt(source_product.apply_inverse(B), product=source_product, return_R=True)
    U_b, s, Vh_b = sp.linalg.svd(R_B.T, full_matrices=False)

    with logger.block(f'Computing generalized left-singular vectors ({modes} vectors) ...'):
        U = Q.lincomb(U_b.T)

    with logger.block(f'Computing generalized right-singular vector ({modes} vectors) ...'):
        Vh = Q_B.lincomb(Vh_b)

    return U[:modes], s[:modes], Vh[:modes]


@defaults('modes', 'p', 'q')
def random_ghep(A, E=None, modes=6, p=20, q=2, single_pass=False):
    r"""Approximates a few eigenvalues of a symmetric linear |Operator| with randomized methods.

    Approximates `modes` eigenvalues `w` with corresponding eigenvectors `v` which solve
    the eigenvalue problem

    .. math::
        A v_i = w_i v_i

    or the generalized eigenvalue problem

    .. math::
        A v_i = w_i E v_i

    if `E` is not `None`.

    This method is an implementation of algorithm 6 and 7 in :cite:`SJK16`.

    Parameters
    ----------
    A
        The Hermitian linear |Operator| for which the eigenvalues are to be computed.
    E
        The Hermitian |Operator| which defines the generalized eigenvalue problem.
    modes
        The number of eigenvalues and eigenvectors which are to be computed.
    p
        If not `0`, adds `p` columns to the randomly sampled matrix in the :func:`rrf` method
        (oversampling parameter).
    q
        If not `0`, performs `q` power iterations to increase the relative weight
        of the larger singular values. Ignored when `single_pass` is `True`.
    single_pass
        If `True`, computes the GHEP where only one set of matvecs Ax is required, but at the
        expense of lower numerical accuracy.
        If `False`, the methods require two sets of matvecs Ax.

    Returns
    -------
    w
        A 1D |NumPy array| which contains the computed eigenvalues.
    V
        A |VectorArray| which contains the computed eigenvectors.
    """
    logger = getLogger('pymor.algorithms.rand_la')

    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range
    assert 0 <= modes <= max(A.source.dim, A.range.dim) and isinstance(modes, int)
    assert 0 <= p <= max(A.source.dim, A.range.dim) - modes and isinstance(p, int)
    assert q >= 0 and isinstance(q, int)

    if E is None:
        E = IdentityOperator(A.source)
    else:
        assert isinstance(E, Operator) and E.linear
        assert not E.parametric
        assert E.source == E.range
        assert E.source == A.source

    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()

    if single_pass:
        Omega = A.source.random(modes+p, distribution='normal')
        Y_bar = A.apply(Omega)
        Y = E.apply_inverse(Y_bar)
        Q, R = gram_schmidt(Y, product=E, return_R=True)
        X = E.apply2(Omega, Q)
        X_lu = lu_factor(X)
        T = lu_solve(X_lu, lu_solve(X_lu, Omega.inner(Y_bar)).T).T
    else:
        C = InverseOperator(E) @ A
        Y, Omega = rrf(C, q=q, l=modes+p, return_rand=True)
        Q = gram_schmidt(Y, product=E)
        T = A.apply2(Q, Q)

    w, S = sp.linalg.eigh(T)
    w = w[::-1]
    S = S[:, ::-1]

    with logger.block(f'Computing eigenvectors ({modes} vectors) ...'):
        V = Q.lincomb(S)

    return w[:modes], V[:modes]
