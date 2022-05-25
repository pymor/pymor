# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import lu_factor, lu_solve
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.operators.interface import Operator
from pymor.core.logger import getLogger
from pymor.operators.constructions import InverseOperator, IdentityOperator


@defaults('tol', 'failure_tolerance', 'num_testvecs')
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
    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)
    assert isinstance(A, Operator)

    B = A.range.empty()

    R = A.source.random(num_testvecs, distribution='normal')
    if iscomplex:
        R += 1j*A.source.random(num_testvecs, distribution='normal')

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:
        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

        def mvinv(v):
            return source_product.apply_inverse(source_product.range.from_numpy(v)).to_numpy()
        L = LinearOperator((source_product.source.dim, source_product.range.dim), matvec=mv)
        Linv = LinearOperator((source_product.range.dim, source_product.source.dim), matvec=mvinv)
        lambda_min = eigsh(L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv)[0]

    testfail = failure_tolerance / min(A.source.dim, A.range.dim)
    testlimit = np.sqrt(2. * lambda_min) * erfinv(testfail**(1. / num_testvecs)) * tol
    maxnorm = np.inf
    M = A.apply(R)

    while(maxnorm > testlimit):
        basis_length = len(B)
        v = A.source.random(distribution='normal')
        if iscomplex:
            v += 1j*A.source.random(distribution='normal')
        B.append(A.apply(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))

    return B


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
    assert isinstance(A, Operator)

    if range_product is None:
        range_product = IdentityOperator(A.range)
    else:
        assert isinstance(range_product, Operator)

    if source_product is None:
        source_product = IdentityOperator(A.source)
    else:
        assert isinstance(source_product, Operator)

    assert 0 <= l <= min(A.source.dim, A.range.dim) and isinstance(l, int)
    assert q >= 0 and isinstance(q, int)

    R = A.source.random(l, distribution='normal')

    if iscomplex:
        R += 1j*A.source.random(l, distribution='normal')

    Q = A.apply(R)
    gram_schmidt(Q, range_product, atol=0, rtol=0, copy=False)

    for i in range(q):
        Q = A.apply_adjoint(range_product.apply(Q))
        Q = source_product.apply_inverse(Q)
        gram_schmidt(Q, source_product, atol=0, rtol=0, copy=False)
        Q = A.apply(Q)
        gram_schmidt(Q, range_product, atol=0, rtol=0, copy=False)

    if return_rand:
        return Q, R
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

    Q = rrf(A, source_product=source_product, range_product=range_product, q=q, l=modes+p)
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
