# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from numbers import Integral, Number

import numpy as np
import scipy as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.special import erfinv

from pymor.algorithms.basic import project_array
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.svd_va import qr_svd
from pymor.core.cache import CacheableObject, cached
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import AdjointOperator, IdentityOperator, InverseOperator
from pymor.operators.interface import Operator
from pymor.tools.deprecated import Deprecated


class RandomizedRangeFinder(CacheableObject):
    """Approximates the range of an |Operator| with randomized methods.

    Parameters
    ----------
    A
        The |Operator| whose range is to be found.
    range_product
        Inner product |Operator| of the range of `A`. Determines the basis orthogonalization and the
        error norm.
    source_product
        Inner product |Operator| of the source of `A`. Determines the basis orthogonalization (only
        if `subspace_iterations` is greater than zero) and the error norm.
    subspace_iterations
        The number of subspace iterations (defaults to zero).
        This can be used to increase the accuracy in the cases where the spectrum of `A` does not
        decay rapidly (at the expense of two additional passes over `A` per subspace iteration).
    lambda_min
        A lower bound for the smallest eigenvalue of `source_product`. Defaults to `None`.
        If `None` and a `source_product` is given, the smallest eigenvalue will be computed
        using :func:`randomized GHEP <pymor.algorithms.rand_la.randomized_ghep>`.
    complex
        If `True`, complex valued random vectors will be chosen.
    self_adjoint
        If `True`, the calculation of the adjoint will be skipped and `op` will be used as its
        adjoint when calculating the subspace iterations. Defaults to `False`.
        Note that when set to `True`, `A` has to be self-adjoint w.r.t. `source_product` and
        `range_product`.
    """

    cache_region = 'memory'

    def __init__(self, A, range_product=None, source_product=None, subspace_iterations=0, lambda_min=None,
                 complex=False, self_adjoint=False):
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

        self.__auto_init(locals())
        self._Q = [self.A.range.empty()]
        for _ in range(subspace_iterations):
            self._Q.append(self.A.source.empty())
            self._Q.append(self.A.range.empty())
        self._Q = tuple(self._Q)
        self._samplevecs = self.A.range.empty()
        self._adjoint_op = A if self_adjoint else AdjointOperator(A, range_product=range_product,
                                                                  source_product=source_product)

    @cached
    def _lambda_min(self):
        if isinstance(self.source_product, IdentityOperator):
            return 1
        elif self.lambda_min is None:
            with self.logger.block('Estimating minimum singular value of source_product ...'):
                return 1/randomized_ghep(InverseOperator(self.source_product), n=1)[0]
        else:
            return self.lambda_min

    def _draw_test_vector(self, n):
        W = self.A.source.random(n, distribution='normal')
        if self.complex:
            W += 1j * self.A.source.random(n, distribution='normal')
        self._samplevecs.append(self.A.apply(W))

    def _estimate_error(self, Q, num_testvecs, p_fail):
        c = np.sqrt(2 * self._lambda_min())
        c *= erfinv((p_fail / min(self.A.source.dim, self.A.range.dim)) ** (1 / num_testvecs))

        if len(self._samplevecs) < num_testvecs:
            self._draw_test_vector(num_testvecs - len(self._samplevecs))

        W = self._samplevecs[:num_testvecs].copy()
        W -= project_array(W, Q, self.range_product)
        return np.max(W.norm(self.range_product)) / c

    def estimate_error(self, basis_size, num_testvecs=20, p_fail=1e-14):
        r"""Randomized a posteriori error estimator for a given basis size.

        This implements the a posteriori error estimator from :cite:`BS18` (Definition 3.1).

        The error estimate is given by

        .. math::
            \epsilon_{\mathrm{est}}=c_{\mathrm{est}}\cdot\max_{\omega\in\Omega}
            \lVert (I-QQ^T)A\omega\rVert_{T\rightarrow S}

        with

        .. math::
            c_{\mathrm{est}}
            =\frac{1}{\sqrt{2\lambda_{\mathrm{min}}}\operatorname{erf}^{-1}
            \left(\left(\frac{\texttt{p_fail}}{n}\right)^{1/\texttt{num_testvecs}}\right)},

        where :math:`\Omega` is a set of `num_testvecs` random vectors, :math:`S` denotes the inner
        product of the range, :math:`T` denotes inner product of the source of :math:`A` and
        :math:`n=\min\{\dim(S),\,\dim(T)\}`.

        Parameters
        ----------
        basis_size
            The size of the basis.
        num_testvecs
            Number of test vectors for error estimation.
        p_fail
            Maximum failure probabilty of the error estimate.

        Returns
        -------
        err
            The approximate error of the basis.
        """
        assert isinstance(basis_size, Integral) and basis_size > 0
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Requested basis is larger than the rank of the operator!')
            self.logger.info('Proceeding with maximum operator rank.')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert 0 < num_testvecs and isinstance(num_testvecs, int)
        assert 0 < p_fail < 1

        Q = self._find_range(basis_size)
        err = self._estimate_error(Q, num_testvecs, p_fail)
        self.logger.info(f'Estimated error (basis dimension {basis_size}): {err:.5e}.')
        return err

    def _extend_basis(self, k):
        W = self.A.source.random(k, distribution='normal')
        if self.complex:
            W += 1j * self.A.source.random(k, distribution='normal')

        offset = len(self._Q[0])
        self._Q[0].append(self.A.apply(W))
        gram_schmidt(self._Q[0], self.range_product, offset=offset, copy=False)

        for i in range(self.subspace_iterations):
            i = 2*i + 1

            k = len(self._Q[i-1]) - offset  # check if GS removed vectors
            offset = len(self._Q[i])
            self._Q[i].append(self._adjoint_op.apply(self._Q[i-1][-k:]))
            gram_schmidt(self._Q[i], self.source_product, offset=offset, copy=False)

            k = len(self._Q[i]) - offset  # check if GS removed vectors
            offset = len(self._Q[i+1])
            self._Q[i+1].append(self.A.apply(self._Q[i][-k:]))
            gram_schmidt(self._Q[i+1], self.range_product, offset=offset, copy=False)

    def _find_range(self, basis_size):
        k = basis_size - len(self._Q[-1])
        if k > 0:
            with self.logger.block(f'Appending {k} basis vector{"s" if k > 1 else ""} ...'):
                self._extend_basis(k)

        k = basis_size - len(self._Q[-1])
        if k > 0:
            self.logger.warning(f'{k} vectors removed in gram_schmidt!')

        return self._Q[-1][:basis_size]

    def find_range(self, basis_size=8, tol=None, num_testvecs=20, p_fail=1e-14, max_basis_size=500):
        r"""Randomized range approximation of the |Operator| `self.A`.

        This method returns a |VectorArray| :math:`Q` whose vectors form an approximate orthonormal
        basis for the range of the |Operator| :math:`A` with the property

        .. math::
            P\left(\epsilon\leq \texttt{tol}\right)\leq1-\texttt{p_fail},

        with

        .. math::
            \epsilon=\lVert A-QQ^TSA\rVert_{T\rightarrow S}=\lVert S^{1/2}(I-QQ^TS)AT^{-1/2}\rVert_2

        where  :math:`S` denotes the inner product of the range and :math:`T` denotes inner product
        of the source of :math:`A`.

        This method employs Algorithm 2 in :cite:`SHB21` with
        :func:`Gram-Schmidt <pymor.algorithms.gram_schmidt.gram_schmidt>` orthogonalization for the
        computation of a basis of size `basis_size`.
        If `tol` is given, the basis will then be adaptively enlarged until the error bound is
        attained. The algorithm for adaptive basis enlargement combines Algorithm 1 in
        :cite:`BS18` with Algorithm 2 in :cite:`SHB21` to incorporate subspace iterations.

        Parameters
        ----------
        basis_size
            The size of the basis that approximates the range. If `tol` is not `None`, this
            can be used to set a lower bound for the dimension of the computed basis.
        tol
            Error tolerance for the computed basis.
        num_testvecs
            Number of test vectors used in `self.estimate_error`.
        p_fail
            Maximum failure probabilty.
        max_basis_size
            Maximum basis size for the adaptive process.

        Returns
        -------
        Q
            |VectorArray| with length greater or equal than `basis_size` which contains an
            approximate basis for the range of `self.A` (with an error bounded by `tol` with
            probability :math:`1-``p_fail`, if supplied).
        """
        assert isinstance(max_basis_size, int) and max_basis_size > 0
        assert isinstance(basis_size, int) and 0 < basis_size
        if basis_size > min(self.A.source.dim, self.A.range.dim):
            self.logger.warning('Requested basis is larger than the rank of the operator!')
            self.logger.info('Proceeding with maximum operator rank.')
            basis_size = min(self.A.source.dim, self.A.range.dim)
        assert tol is None or tol > 0
        assert isinstance(num_testvecs, int) and num_testvecs > 0
        assert 0 < p_fail < 1

        with self.logger.block('Finding range ...'):
            with self.logger.block(f'Approximating range basis of dimension {basis_size} ...'):
                self._find_range(basis_size)
                err = self._estimate_error(self._Q[-1], num_testvecs, p_fail)

            if tol is not None and err > tol:
                with self.logger.block('Extending range basis adaptively ...'):
                    max_iter = min(max_basis_size, self.A.source.dim, self.A.range.dim)
                    while len(self._Q[-1]) < max_iter:
                        basis_size = min(basis_size + 1, max_iter)
                        err = self._estimate_error(self._Q[-1], num_testvecs, p_fail)
                        self.logger.info(f'Basis dimension: {basis_size}/{max_iter}\t'
                                         + 'Estimated error: {err:.5e} (tol={tol:.2e})')
                        if err <= tol:
                            break

        Q = self._Q[-1][:basis_size]
        self.logger.info(f'Found range of dimension {len(Q)}. (Estimated error: {err:.5e})')

        return Q

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
        return Q, RRF._samplevecs
    else:
        return Q


@defaults('p', 'q', 'modes')
@Deprecated('randomized_svd')
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
    return randomized_svd(A, modes, range_product=range_product, source_product=source_product,
                          subspace_iterations=q, oversampling=p)


@defaults('oversampling', 'subspace_iterations')
def randomized_svd(A, n, range_product=None, source_product=None, subspace_iterations=0, oversampling=20):
    r"""Randomized SVD of an |Operator| based on :cite:`SHB21`.

    Viewing the |Operator| :math:`A` as an :math:`m` by :math:`n` matrix, this methods computes and
    returns the randomized generalized singular value decomposition of `A` according :cite:`SHB21`:

    .. math::

        A = U \Sigma V^{-1},

    where the inner product on the range :math:`\mathbb{R}^m` is given by

    .. math::

        (x, y)_S = x^TSy

    and the inner product on the source :math:`\mathbb{R}^n` is given by

    .. math::

        (x, y)_T = x^TTy.

    Note that :math:`U` is :math:`S`-orthogonal, i.e. :math:`U^TSU=I`
    and :math:`V` is :math:`T`-orthogonal, i.e. :math:`V^TTV=I`.
    In particular, `V^{-1}=V^TT`.

    Parameters
    ----------
    A
        The |Operator| for which the randomized SVD is to be computed.
    n
        The number of eigenvalues and eigenvectors which are to be computed.
    range_product
        Range product |Operator| :math:`S` w.r.t. which the randomized SVD is computed.
    source_product
        Source product |Operator| :math:`T` w.r.t. which the randomized SVD is computed.
    subspace_iterations
        The number of subspace iterations to increase the relative weight
        of the larger singular values (ignored when `single_pass` is `True`).
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

    RRF = RandomizedRangeFinder(A, subspace_iterations=subspace_iterations, range_product=range_product,
                                source_product=source_product)

    assert 0 <= n <= max(A.source.dim, A.range.dim) and isinstance(n, int)
    assert 0 <= oversampling and isinstance(oversampling, int)
    if oversampling > max(A.source.dim, A.range.dim) - n:
        logger.warning('Oversampling parameter is too large!')
        oversampling = max(A.source.dim, A.range.dim) - n
        logger.info(f'Setting oversampling to {oversampling} and proceeding ...')

    if range_product is None:
        range_product = IdentityOperator(A.range)
    if source_product is None:
        source_product = IdentityOperator(A.source)
    if A.source.dim == 0 or A.range.dim == 0:
        return A.source.empty(), np.array([]), A.range.empty()

    with logger.block('Approximating basis for the operator range ...'):
        Q = RRF.find_range(basis_size=n+oversampling)

    with logger.block(f'Computing transposed SVD in the reduced space ({len(Q)}x{Q.dim})...'):
        B = source_product.apply_inverse(A.apply_adjoint(range_product.apply(Q)))
        V, s, Uh_b = qr_svd(B, product=source_product, modes=n, rtol=0)

    with logger.block('Backprojecting the left'
                      + f'{" " if isinstance(range_product, IdentityOperator) else " generalized "}'
                      + f'singular vector{"s" if n > 1 else ""} ...'):
        U = Q.lincomb(Uh_b[:n])

    return U, s, V


@defaults('modes', 'p', 'q')
@Deprecated('randomized_ghep')
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
    return randomized_ghep(A, E=E, n=modes, subspace_iterations=q, oversampling=p, single_pass=single_pass,
                           return_evecs=True)


@defaults('n', 'oversampling', 'subspace_iterations')
def randomized_ghep(A, E=None, n=6, subspace_iterations=0, oversampling=20, single_pass=False, return_evecs=False):
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
    subspace_iterations
        The number of subspace iterations to increase the relative weight
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

    assert isinstance(A, Operator) and A.linear
    assert not A.parametric
    assert A.source == A.range
    assert 0 <= n <= max(A.source.dim, A.range.dim) and isinstance(n, int)
    assert 0 <= oversampling and isinstance(oversampling, int)
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
            X_lu = lu_factor(X)
            T = lu_solve(X_lu, lu_solve(X_lu, W.inner(Y_bar)).T).T
    else:
        with logger.block('Approximating basis for the operator source/range ...'):
            C = InverseOperator(E) @ A
            RRF = RandomizedRangeFinder(C, subspace_iterations=subspace_iterations, range_product=E, source_product=E,
                                        self_adjoint=True)
            Q = RRF.find_range(n+oversampling)
        with logger.block('Projecting operator onto the reduced space ...'):
            T = A.apply2(Q, Q)

    if return_evecs:
        with logger.block(f'Computing the{" " if isinstance(E, IdentityOperator) else " generalized "}'
                          + f'eigenvalue{"s" if n > 1 else ""} and eigenvector{"s" if n > 1 else ""} '
                          + 'in the reduced space ...'):
            w, Vr = sp.linalg.eigh(T, subset_by_index=(T.shape[1]-n, T.shape[0]-1))
        with logger.block('Backprojecting the'
                          + f'{" " if isinstance(E, IdentityOperator) else " generalized "}'
                          + f'eigenvector{"s" if n > 1 else ""} ...'):
            V = Q.lincomb(Vr[:, ::-1].T)
        return w[::-1], V
    else:
        with logger.block(f'Computing the{" " if isinstance(E, IdentityOperator) else " generalized "}'
                          + f'eigenvalue{"s" if n > 1 else ""} in the reduced space ...'):
            return sp.linalg.eigh(T, subset_by_index=(T.shape[1]-n, T.shape[0]-1), eigvals_only=True)[::-1]
