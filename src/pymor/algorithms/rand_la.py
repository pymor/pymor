# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Number
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
from pymor.vectorarrays.numpy import NumpyVectorSpace


class RandomizedRangeFinder(CacheableObject):
    def __init__(self, A, subspace_iterations=0, range_product=None, source_product=None, lambda_min=None,
                 complex=False):
        assert isinstance(A, Operator)
        assert 0 <= subspace_iterations and isinstance(subspace_iterations, int)
        if range_product is None:
            range_product = IdentityOperator(A.range)

        if source_product is None:
            source_product = IdentityOperator(A.source)
        assert isinstance(range_product, Operator)
        assert source_product.source == source_product.range == A.source
        assert isinstance(source_product, Operator)
        assert source_product.source == source_product.range == A.source
        assert lambda_min is None or isinstance(lambda_min, Number)
        assert isinstance(complex, bool)

        self.__auto_init(locals())
        self._Q = [self.A.range.empty()]
        for _ in range(subspace_iterations):
            self._Q.append(self.A.source.empty())
            self._Q.append(self.A.range.empty())
        self._Q = tuple(self._Q)
        self.testvecs = self.A.source.empty()

    @cached
    def _lambda_min(self):
        if isinstance(self.source_product, IdentityOperator):
            return 1
        elif self.lambda_min is None:
            with self.logger.block('Estimating minimum singular value of source_product ...'):
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
        W = self.A.source.random(n, distribution='normal')
        if self.complex:
            W += 1j * self.A.source.random(n, distribution='normal')
        self.testvecs.append(self.A.apply(W))

    def _estimate_error(self, basis_size, num_testvecs, p_fail):
        c = np.sqrt(2 * self._lambda_min())
        c *= erfinv((p_fail / min(self.A.source.dim, self.A.range.dim)) ** (1 / num_testvecs))

        if len(self.testvecs) < num_testvecs:
            self._draw_test_vector(num_testvecs - len(self.testvecs))

        W, Q = self.testvecs[:num_testvecs].copy(), self._find_range(basis_size)
        W -= Q.lincomb(Q.inner(W, self.range_product).T)
        return c * np.max(W.norm(self.range_product))

    def estimate_error(self, basis_size, num_testvecs=20, p_fail=1e-14):
        assert isinstance(basis_size, int) and basis_size > 0
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Requested basis is larger than the rank of the operator!')
            self.logger.info('Proceeding with maximum operator rank.')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert 0 < num_testvecs and isinstance(num_testvecs, int)
        assert 0 < p_fail

        err = self._estimate_error(basis_size, num_testvecs, p_fail)
        self.logger.info(f'Estimated error (basis dimension {basis_size}): {err:.5e}.')
        return err

    def _extend_basis(self, k):
        W = self.A.source.random(k, distribution='normal')
        if self.complex:
            W += 1j * self.A.source.random(k, distribution='normal')

        self._Q[0].append(self.A.apply(W))
        gram_schmidt(self._Q[0], self.range_product, offset=len(self._Q[0]), copy=False)

        for i in range(self.subspace_iterations):
            i = 2*i + 1
            self._Q[i].append(self.source_product.apply_inverse(
                (self.A.apply_adjoint(self.range_product.apply(self._Q[i-1][-k:])))))
            gram_schmidt(self._Q[i], self.source_product, offset=len(self._Q[i]), copy=False)
            self._Q[i+1].append(self.A.apply(self._Q[i][-k:]))
            gram_schmidt(self._Q[i+1], self.range_product, offset=len(self._Q[i+1]), copy=False)

    def _find_range(self, basis_size):
        if basis_size > len(self._Q[-1]):
            k = basis_size - len(self._Q[-1])
            with self.logger.block(f'Appending {k} basis vector{"s" if k > 1 else ""} ...'):
                self._extend_basis(k)
            while basis_size > len(self._Q[-1]):
                k = basis_size - len(self._Q[-1])
                with self.logger.block(f'Appending {k} basis vector{"s" if k > 1 else ""}'
                                       + 'to compensate for removal in gram_schmidt ...'):
                    self._extend_basis(k)

        return self._Q[-1][:basis_size]

    def find_range(self, basis_size=8, tol=None, num_testvecs=20, p_fail=1e-14, block_size=8,
                   increase_block=True, max_basis_size=500):
        assert isinstance(basis_size, int) and basis_size > 0
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Requested basis is larger than the rank of the operator!')
            self.logger.info('Proceeding with maximum operator rank.')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert tol is None or tol > 0
        assert isinstance(num_testvecs, int) and num_testvecs > 0
        assert p_fail > 0

        with self.logger.block('Finding range ...'):
            with self.logger.block(f'Approximating range basis of dimension {basis_size} ...'):
                self._find_range(basis_size)
                err = self._estimate_error(basis_size, num_testvecs, p_fail)

            if tol is not None and err > tol:
                with self.logger.block('Extending range basis adaptively ...'):
                    max_iter = min(max_basis_size, self.A.source.dim, self.A.range.dim)
                    while len(self._Q[-1]) < max_iter:
                        basis_size = min(basis_size + 1, max_iter)
                        err = self._estimate_error(basis_size, num_testvecs, p_fail)
                        self.logger.info(f'Basis dimension: {basis_size}/{max_iter}\t'
                                         + 'Estimated error: {err:.5e} (tol={tol:.2e})')
                        if err <= tol:
                            break

        self.logger.info(f'Found range of dimension {basis_size}. (Estimated error: {err:.5e})')

        return self._find_range(basis_size)


@defaults('tol', 'failure_tolerance', 'num_testvecs')
@Deprecated('RandomizedRangeFinder')
def adaptive_rrf(A, range_product=None, source_product=None, tol=1e-4,
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
    range_product
        Inner product |Operator| of the range of A.
    source_product
        Inner product |Operator| of the source of A.
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
    RRF = RandomizedRangeFinder(A, subspace_iterations=0, range_product=range_product, source_product=source_product,
                                lambda_min=lambda_min, complex=iscomplex)
    return RRF.find_range(basis_size=1, tol=tol, num_testvecs=num_testvecs, p_fail=failure_tolerance)


@defaults('q', 'l')
@Deprecated('RandomizedRangeFinder')
def rrf(A, range_product=None, source_product=None, q=2, l=8, return_rand=False, iscomplex=False):
    r"""Randomized range approximation of `A`.

    Given the |Operator| `A`, the return value of this method is the |VectorArray|
    `Q` whose vectors form an approximate orthonormal basis for the range of `A`.

    This method is based on algorithm 2 in :cite:`SHB21`.

    Parameters
    ----------
    A
        The |Operator| A.
    range_product
        Inner product |Operator| of the range of A.
    source_product
        Inner product |Operator| of the source of A.
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
    RRF = RandomizedRangeFinder(A, subspace_iterations=q, range_product=range_product, source_product=source_product,
                                complex=iscomplex)
    Q = RRF.find_range(basis_size=l, tol=None)
    if return_rand:
        return Q, RRF.testvecs
    else:
        return Q


@defaults('oversampling', 'subspace_iterations')
def randomized_svd(A, n, range_product=None, source_product=None, oversampling=20, subspace_iterations=2):
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
    n
        The number of eigenvalues and eigenvectors which are to be computed.
    range_product
        Range product |Operator| :math:`S` w.r.t which the randomized SVD is computed.
    source_product
        Source product |Operator| :math:`T` w.r.t which the randomized SVD is computed.
    oversampling
        The number of samples that are drawn in addition to the desired basis size in the
        randomized range approximation process.
    subspace_iterations
        The number of subspace iterations to increase the relative weight
        of the larger singular values. Ignored when `single_pass` is `True`.

    Returns
    -------
    U
        |VectorArray| of approximated left singular vectors.
    s
        One-dimensional |NumPy array| of the approximated singular values.
    V
        |VectorArray| of the approximated right singular vectors.
    """
    logger = getLogger('pymor.algorithms.rand_la.randomized_svd')

    RRF = RandomizedRangeFinder(A, subspace_iterations=subspace_iterations, range_product=range_product,
                                source_product=source_product)

    assert 0 <= n <= max(A.source.dim, A.range.dim) and isinstance(n, int)
    assert 0 <= oversampling <= max(A.source.dim, A.range.dim) - n and isinstance(oversampling, int)
    if range_product is None:
        range_product = IdentityOperator(A.range)
    if source_product is None:
        source_product = IdentityOperator(A.source)
    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()

    with logger.block('Approximating basis for the operator range ...'):
        Q = RRF.find_range(basis_size=n+oversampling)

    with logger.block('Projecting operator onto the reduced space ...'):
        if isinstance(source_product, IdentityOperator):
            R_B = A.apply_adjoint(Q).to_numpy().T
        else:
            B = A.apply_adjoint(range_product.apply(Q))
            Q_B, R_B = gram_schmidt(source_product.apply_inverse(B), product=source_product, return_R=True)

    with logger.block(f'Computing SVD in the reduced space ({R_B.shape[1]}x{R_B.shape[0]})...'):
        U_b, s, Vh_b = sp.linalg.svd(R_B.T, full_matrices=False)

    with logger.block('Backprojecting the left'
                      + f'{" " if isinstance(range_product, IdentityOperator) else " generalized "}'
                      + 'singular vectors...'):
        U = Q.lincomb(U_b[:, :n].T)

    if isinstance(source_product, IdentityOperator):
        V = NumpyVectorSpace.from_numpy(Vh_b[:n])
    else:
        with logger.block('Backprojecting the right generalized singular vectors ...'):
            V = Q_B.lincomb(Vh_b[:n])

    return U, s[:n], V


@defaults('n', 'oversampling', 'subspace_iterations')
def randomized_ghep(A, E=None, n=6, oversampling=20, subspace_iterations=2, single_pass=False, return_evecs=False):
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
    n
        The number of eigenvalues and eigenvectors which are to be computed.
    oversampling
        The number of samples that are drawn in addition to the desired basis size in the
        randomized range approximation process.
    subspace_iterations
        The number of subspace iterations to increase the relative weight
        of the larger singular values. Ignored when `single_pass` is `True`.
    single_pass
        If `True`, computes the GHEP where only one set of matvecs Ax is required, but at the
        expense of lower numerical accuracy.
        If `False`, the methods performs two sets of matvecs Ax.
    return_evecs
        If `True`, the eigenvectors are computed and returned.

    Returns
    -------
    w
        A 1D |NumPy array| which contains the computed eigenvalues.
    V
        A |VectorArray| which contains the computed eigenvectors.
    """
    logger = getLogger('pymor.algorithms.rand_la.randomized_ghep')

    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range
    assert 0 <= n <= max(A.source.dim, A.range.dim) and isinstance(n, int)
    assert 0 <= oversampling <= max(A.source.dim, A.range.dim) - n and isinstance(oversampling, int)
    assert subspace_iterations >= 0 and isinstance(subspace_iterations, int)
    assert isinstance(single_pass, bool)
    assert isinstance(return_evecs, bool)

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
        W = A.source.random(n+oversampling, distribution='normal')
        Y_bar = A.apply(W)
        Y = E.apply_inverse(Y_bar)
        Q = gram_schmidt(Y, product=E)
        X = E.apply2(W, Q)
        X_lu = lu_factor(X)
        T = lu_solve(X_lu, lu_solve(X_lu, W.inner(Y_bar)).T).T
    else:
        C = InverseOperator(E) @ A
        RRF = RandomizedRangeFinder(C, subspace_iterations=subspace_iterations, range_product=E)
        Q = RRF.find_range(n+oversampling)
        T = A.apply2(Q, Q)

    w, *S = sp.linalg.eigh(T, evals_only=not return_evecs)
    w = w[::-1]
    if return_evecs:
        with logger.block(f'Computing eigenvectors ({n} vectors) ...'):
            S = S[0][:, ::-1]
            V = Q.lincomb(S)
        return w[:n], V[:n]
    else:
        return w[:n]
