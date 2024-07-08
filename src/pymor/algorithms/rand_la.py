# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
from scipy.special import erfinv

from pymor.algorithms.eigs import eigs
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.svd_va import qr_svd
from pymor.core.base import BasicObject
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import AdjointOperator, IdentityOperator, InverseOperator
from pymor.operators.interface import Operator


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
    A_adj
        Adjoint |Operator| to use for power iterations. If `None` the
        adjoint is computed using `A`, `source_product` and `range_product`.
        Set to `A` for a `self` for a known self-adjoint operator.
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
                self.lambda_min = eigs(source_product, sigma=0, which='LM', k=1)[0][0].real

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


@defaults('oversampling', 'power_iterations')
def randomized_svd(A, n, source_product=None, range_product=None, power_iterations=0, oversampling=20):
    r"""Randomized SVD of an |Operator| based on :cite:`SHB21`.

    Viewing the |Operator| :math:`A` as an :math:`m` by :math:`n` matrix, this methods computes and
    returns the randomized generalized singular value decomposition of `A` according :cite:`SHB21`:

    .. math::

        A = U \Sigma V^{-1},

    where the inner product on the range :math:`\mathbb{R}^m` is given by

    .. math::

        (x, y)_R = x^TRy

    and the inner product on the source :math:`\mathbb{R}^n` is given by

    .. math::

        (x, y)_S = x^TSy.

    Note that :math:`U` is :math:`R`-orthogonal, i.e. :math:`U^TRU=I`
    and :math:`V` is :math:`S`-orthogonal, i.e. :math:`V^TSV=I`.
    In particular, `V^{-1}=V^TS`.

    Parameters
    ----------
    A
        The |Operator| for which the randomized SVD is to be computed.
    n
        The number of eigenvalues and eigenvectors which are to be computed.
    source_product
        Source product |Operator| :math:`S` w.r.t. which the randomized SVD is computed.
    range_product
        Range product |Operator| :math:`R` w.r.t. which the randomized SVD is computed.
    power_iterations
        The number of power iterations to increase the relative weight of the larger singular
        values.
    oversampling
        The number of samples that are drawn in addition to the desired basis size in the
        randomized range approximation process.

    Returns
    -------
    U
        |VectorArray| containing the approximated left singular vectors.
    s
        One-dimensional |NumPy array| of the approximated singular values.
    V
        |VectorArray| containing the approximated right singular vectors.
    """
    logger = getLogger('pymor.algorithms.rand_la.randomized_svd')

    RRF = RandomizedRangeFinder(A, power_iterations=power_iterations, range_product=range_product,
                                source_product=source_product)

    assert 0 <= n <= max(A.source.dim, A.range.dim)
    assert 0 <= oversampling
    if oversampling > max(A.source.dim, A.range.dim) - n:
        logger.warning('Oversampling parameter is too large!')
        oversampling = max(A.source.dim, A.range.dim) - n
        logger.info(f'Setting oversampling to {oversampling} and proceeding ...')

    if range_product is None:
        range_product = IdentityOperator(A.range)
    if source_product is None:
        source_product = IdentityOperator(A.source)

    assert isinstance(range_product, Operator)
    assert range_product.source == range_product.range == A.range
    assert isinstance(source_product, Operator)
    assert source_product.source == source_product.range == A.source

    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()

    with logger.block('Approximating basis for the operator range ...'):
        Q = RRF.find_range(basis_size=n+oversampling)

    with logger.block(f'Computing transposed SVD in the reduced space ({len(Q)}x{Q.dim})...'):
        B = source_product.apply_inverse(A.apply_adjoint(range_product.apply(Q)))
        V, s, Uh_b = qr_svd(B, product=source_product, modes=n, rtol=0)

    with logger.block('Backprojecting the left'
                      f'{" " if isinstance(range_product, IdentityOperator) else " generalized "}'
                      f'singular vector{"s" if n > 1 else ""} ...'):
        U = Q.lincomb(Uh_b[:n])

    return U, s, V


@defaults('n', 'oversampling', 'power_iterations')
def randomized_ghep(A, E=None, n=6, power_iterations=0, oversampling=20, single_pass=False, return_evecs=False):
    r"""Approximates a few eigenvalues of a Hermitian linear |Operator| with randomized methods.

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
        The number of eigenvalues and eigenvectors which are to be computed. Defaults to 6.
    power_iterations
        The number of power iterations to increase the relative weight
        of the larger singular values. Ignored when `single_pass` is `True`.
    oversampling
        The number of samples that are drawn in addition to the desired basis size in the
        randomized range approximation process.
    single_pass
        If `True`, computes the GHEP where only one set of matvecs Ax is required, but at the
        expense of lower numerical accuracy.
        If `False`, the methods performs two sets of matvecs Ax (default).
    return_evecs
        If `True`, the eigenvectors are computed and returned. Defaults to `False`.

    Returns
    -------
    w
        A 1D |NumPy array| which contains the computed eigenvalues.
    V
        A |VectorArray| which contains the computed eigenvectors.
    """
    logger = getLogger('pymor.algorithms.rand_la.randomized_ghep')

    assert isinstance(A, Operator)
    assert A.linear
    assert not A.parametric
    assert A.source == A.range
    assert 0 <= n <= max(A.source.dim, A.range.dim)
    assert 0 <= oversampling
    assert power_iterations >= 0

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

    if oversampling > max(A.source.dim, A.range.dim) - n:
        logger.warning('Oversampling parameter is too large!')
        oversampling = max(A.source.dim, A.range.dim) - n
        logger.info(f'Setting oversampling to {oversampling} and proceeding ...')

    if single_pass:
        with logger.block('Approximating basis for the operator source/range (single pass) ...'):
            W = A.source.random(n+oversampling, distribution='normal')
            Y_bar = A.apply(W)
            Y = E.apply_inverse(Y_bar)
            Q = gram_schmidt(Y, product=E)
        with logger.block('Projecting operator onto the reduced space ...'):
            X = E.apply2(W, Q)
            X_lu = spla.lu_factor(X)
            T = spla.lu_solve(X_lu, spla.lu_solve(X_lu, W.inner(Y_bar)).T).T
    else:
        with logger.block('Approximating basis for the operator source/range ...'):
            C = InverseOperator(E) @ A
            RRF = RandomizedRangeFinder(C, power_iterations=power_iterations, range_product=E, source_product=E,
                                        A_adj=C)
            Q = RRF.find_range(n+oversampling)
        with logger.block('Projecting operator onto the reduced space ...'):
            T = A.apply2(Q, Q)

    if return_evecs:
        with logger.block(f'Computing the{" " if isinstance(E, IdentityOperator) else " generalized "}'
                          f'eigenvalue{"s" if n > 1 else ""} and eigenvector{"s" if n > 1 else ""} '
                          'in the reduced space ...'):
            w, Vr = spla.eigh(T, subset_by_index=(T.shape[1]-n, T.shape[0]-1))
        with logger.block('Backprojecting the'
                          f'{" " if isinstance(E, IdentityOperator) else " generalized "}'
                          f'eigenvector{"s" if n > 1 else ""} ...'):
            V = Q.lincomb(Vr[:, ::-1].T)
        return w[::-1], V
    else:
        with logger.block(f'Computing the{" " if isinstance(E, IdentityOperator) else " generalized "}'
                          f'eigenvalue{"s" if n > 1 else ""} in the reduced space ...'):
            return spla.eigh(T, subset_by_index=(T.shape[1]-n, T.shape[0]-1), eigvals_only=True)[::-1]
