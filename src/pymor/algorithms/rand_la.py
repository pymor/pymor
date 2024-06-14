# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve, svd
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import AdjointOperator, IdentityOperator, InverseOperator
from pymor.operators.interface import Operator
from pymor.tools.deprecated import Deprecated


class RandomizedRangeFinder(BasicObject):
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
    power_iterations
        Number of power iterations.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    block_size
        Number of basis vectors to add per iteration.
    iscomplex
        If `True`, the random vectors are chosen complex.
    """

    @defaults('num_testvecs', 'failure_tolerance')
    def __init__(self, A, source_product=None, range_product=None, A_adj=None,
                 power_iterations=0, failure_tolerance=1e-15, num_testvecs=20,
                 lambda_min=None, block_size=None, iscomplex=False):
        assert source_product is None or isinstance(source_product, Operator)
        assert range_product is None or isinstance(range_product, Operator)
        assert isinstance(A, Operator)
        assert lambda_min is None or lambda_min > 0

        if A_adj is None:
            A_adj = AdjointOperator(A, range_product=range_product, source_product=source_product)

        self.__auto_init(locals())
        self.R, self.estimator_last_basis_size, self.last_estimated_error = None, 0, np.inf
        self.Q = [A.range.empty() for _ in range(power_iterations+1)]

    def estimate_error(self):
        A, range_product, num_testvecs = self.A, self.range_product, self.num_testvecs

        if self.lambda_min is None:
            source_product = self.source_product
            if source_product is None:
                self.lambda_min = 1
            else:
                assert source_product is not None
                def mv(v):
                    return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

                def mvinv(v):
                    return source_product.apply_inverse(source_product.range.from_numpy(v)).to_numpy()
                L = LinearOperator((source_product.source.dim, source_product.range.dim), matvec=mv)
                Linv = LinearOperator((source_product.range.dim, source_product.source.dim), matvec=mvinv)
                self.lambda_min = eigsh(L, sigma=0, which='LM', return_eigenvectors=False, k=1, OPinv=Linv)[0]

        if self.R is None:
            Omega_test = A.source.random(num_testvecs, distribution='normal')
            if self.iscomplex:
                Omega_test += 1j*A.source.random(num_testvecs, distribution='normal')
            self.R = A.apply(Omega_test)

        if len(self.Q[-1]) > self.estimator_last_basis_size:
            # in an older implementation, we used re-orthogonalization here, i.e,
            # projecting onto Q[-1] instead of new_basis_vecs
            # should not be needed in most cases. add an option?
            new_basis_vecs = self.Q[-1][self.estimator_last_basis_size:]
            self.R -= new_basis_vecs.lincomb(new_basis_vecs.inner(self.R, product=range_product).T)
            self.estimator_last_basis_size += len(new_basis_vecs)

        testfail = self.failure_tolerance / min(A.source.dim, A.range.dim)
        testlimit = np.sqrt(2. * self.lambda_min) * erfinv(testfail**(1. / num_testvecs))
        maxnorm = np.max(self.R.norm(range_product))
        self.last_estimated_error = maxnorm / testlimit
        return self.last_estimated_error

    def find_range(self, basis_size=None, tol=None):
        """Find the range of A.

        Parameters
        ----------
        basis_size
            Maximum dimension of range approximation.
        tol
            Error tolerance for the algorithm.

        Returns
        -------
        |VectorArray| which contains the basis, whose span approximates the range of A.
        """
        A, A_adj, Q, range_product = self.A, self.A_adj, self.Q, self.range_product

        if basis_size is None and tol is None:
            raise ValueError('Must specify basis_size or tol.')

        if basis_size is not None and basis_size <= len(Q[-1]):
            self.logger.info('Smaller basis size requested than already computed.')
            return Q[-1][:basis_size].copy()

        if tol is not None and tol >= self.last_estimated_error:
            self.logger.info('Tolerance larger than last estimated error. Returning existing basis.')
            return Q[-1].copy()


        while True:
            # termination criteria
            if basis_size is not None and basis_size <= len(Q[-1]):
                self.logger.info('Prescribed basis size reached.')
                break

            if tol is not None:  # error estimator is only evaluated when needed
                estimated_error = self.estimate_error()
                self.logger.info(f'Estimated error: {estimated_error}')
                if estimated_error < tol:
                    self.logger.info('Prescribed error tolerance reached.')
                    break

            # compute new basis vectors
            block_size = (self.block_size if self.block_size is not None else
                          1 if tol is not None else
                          basis_size)
            if basis_size is not None:
                block_size = min(block_size, basis_size - len(Q[-1]))

            V = A.source.random(block_size, distribution='normal')
            if self.iscomplex:
                V += 1j*A.source.random(block_size, distribution='normal')

            current_len = len(Q[0])
            Q[0].append(A.apply(V))
            gram_schmidt(Q[0], range_product, atol=0, rtol=0, offset=current_len, copy=False)
            if len(Q[0]) == current_len:
                raise ValueError('Basis extension broke down before convergence.')

            # power iterations
            for i in range(1, len(Q)):
                V = Q[i-1][current_len:]
                current_len = len(Q[i])
                Q[i].append(A.apply(A_adj.apply(V)))
                gram_schmidt(Q[i], range_product, atol=0, rtol=0, offset=current_len, copy=False)
                if len(Q[i]) == current_len:
                    raise ValueError('Basis extension broke down before convergence.')

        if basis_size is not None and basis_size < len(Q[-1]):
            return Q[-1][:basis_size].copy()
        else:
            # special case to avoid deep copy of array
            return Q[-1].copy()


@Deprecated(RandomizedRangeFinder)
@defaults('tol', 'failure_tolerance', 'num_testvecs')
def adaptive_rrf(A, source_product=None, range_product=None, tol=1e-4,
                 failure_tolerance=1e-15, num_testvecs=20, lambda_min=None, iscomplex=False):
    return RandomizedRangeFinder(A, source_product=source_product, range_product=range_product,
                                 failure_tolerance=failure_tolerance, num_testvecs=num_testvecs,
                                 lambda_min=lambda_min, iscomplex=iscomplex).find_range(tol=tol)


@Deprecated(RandomizedRangeFinder)
@defaults('q', 'l')
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
    if return_rand:
        raise NotImplementedError('No longer available.')
    return RandomizedRangeFinder(A, source_product=source_product, range_product=range_product,
                                 power_iterations=q, iscomplex=iscomplex).find_range(basis_size=l)


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

    assert 0 <= modes <= max(A.source.dim, A.range.dim)
    assert isinstance(modes, int)
    assert 0 <= p <= max(A.source.dim, A.range.dim) - modes
    assert isinstance(p, int)
    assert q >= 0
    assert isinstance(q, int)

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

    Q = rrf(A, source_product=source_product, range_product=range_product, q=q, l=modes+p)
    B = A.apply_adjoint(range_product.apply(Q))
    Q_B, R_B = gram_schmidt(source_product.apply_inverse(B), product=source_product, return_R=True)
    U_b, s, Vh_b = svd(R_B.T, full_matrices=False)

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

    assert isinstance(A, Operator)
    assert A.linear
    assert not A.parametric
    assert A.source == A.range
    assert 0 <= modes <= max(A.source.dim, A.range.dim)
    assert isinstance(modes, int)
    assert 0 <= p <= max(A.source.dim, A.range.dim) - modes
    assert isinstance(p, int)
    assert q >= 0
    assert isinstance(q, int)

    if E is None:
        E = IdentityOperator(A.source)
    else:
        assert isinstance(E, Operator)
        assert E.linear
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
        Y = rrf(C, q=q, l=modes+p)
        Q = gram_schmidt(Y, product=E)
        T = A.apply2(Q, Q)

    w, S = eigh(T)
    w = w[::-1]
    S = S[:, ::-1]

    with logger.block(f'Computing eigenvectors ({modes} vectors) ...'):
        V = Q.lincomb(S)

    return w[:modes], V[:modes]
