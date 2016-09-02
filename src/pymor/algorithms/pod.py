# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.linalg import eigh

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.floatcmp import float_cmp_all
from pymor.vectorarrays.interfaces import VectorArrayInterface


@defaults('rtol', 'atol', 'l2_mean_err', 'symmetrize', 'orthonormalize', 'check', 'check_tol')
def pod(A, modes=None, product=None, rtol=4e-8, atol=0., l2_mean_err=0.,
        symmetrize=False, orthonormalize=True, check=True, check_tol=1e-10):
    """Proper orthogonal decomposition of `A`.

    Viewing the |VectorArray| `A` as a `A.dim` x `len(A)` matrix,
    the return value of this method is the |VectorArray| of left-singular
    vectors of the singular value decomposition of `A`, where the inner product
    on R^(`dim(A)`) is given by `product` and the inner product on R^(`len(A)`)
    is the Euclidean inner product.

    Parameters
    ----------
    A
        The |VectorArray| for which the POD is to be computed.
    modes
        If not `None`, only the first `modes` POD modes (singular vectors) are
        returned.
    product
        Inner product |Operator| w.r.t. which the POD is computed.
    rtol
        Singular values smaller than this value multiplied by the largest singular
        value are ignored.
    atol
        Singular values smaller than this value are ignored.
    l2_mean_err
        Do not return more modes than needed to bound the mean l2-approximation
        error by this value. I.e. the number of returned modes is at most ::

            argmin_N { 1 / len(A) * sum_{n=N+1}^{infty} s_n^2 <= l2_mean_err^2 }

        where `s_n` denotes the n-th singular value.
    symmetrize
        If `True`, symmetrize the Gramian again before proceeding.
    orthonormalize
        If `True`, orthonormalize the computed POD modes again using
        the :func:`~pymor.algorithms.gram_schmidt.gram_schmidt` algorithm.
    check
        If `True`, check the computed POD modes for orthonormality.
    check_tol
        Tolerance for the orthonormality check.

    Returns
    -------
    POD
        |VectorArray| of POD modes.
    SVALS
        Sequence of singular values.
    """

    assert isinstance(A, VectorArrayInterface)
    assert len(A) > 0
    assert modes is None or modes <= len(A)
    assert product is None or isinstance(product, OperatorInterface)

    logger = getLogger('pymor.algorithms.pod.pod')

    with logger.block('Computing Gramian ({} vectors) ...'.format(len(A))):
        B = A.gramian() if product is None else product.apply2(A, A)

        if symmetrize:     # according to rbmatlab this is necessary due to rounding
            B = B + B.T
            B *= 0.5

    with logger.block('Computing eigenvalue decomposition ...'):
        eigvals = None if (modes is None or l2_mean_err > 0.) else (len(B) - modes, len(B) - 1)

        EVALS, EVECS = eigh(B, overwrite_a=True, turbo=True, eigvals=eigvals)
        EVALS = EVALS[::-1]
        EVECS = EVECS.T[::-1, :]  # is this a view? yes it is!

        tol = max(rtol ** 2 * EVALS[0], atol ** 2)
        above_tol = np.where(EVALS >= tol)[0]
        if len(above_tol) == 0:
            return A.space.empty(), np.array([])
        last_above_tol = above_tol[-1]

        errs = np.concatenate((np.cumsum(EVALS[::-1])[::-1], [0.]))
        below_err = np.where(errs <= l2_mean_err**2 * len(A))[0]
        first_below_err = below_err[0]

        selected_modes = min(first_below_err, last_above_tol + 1)
        if modes is not None:
            selected_modes = min(selected_modes, modes)

        SVALS = np.sqrt(EVALS[:selected_modes])
        EVECS = EVECS[:selected_modes]

    with logger.block('Computing left-singular vectors ({} vectors) ...'.format(len(EVECS))):
        POD = A.lincomb(EVECS / SVALS[:, np.newaxis])

    if orthonormalize:
        with logger.block('Re-orthonormalizing POD modes ...'):
            POD = gram_schmidt(POD, product=product, copy=False)

    if check:
        logger.info('Checking orthonormality ...')
        if not product and not float_cmp_all(POD.dot(POD), np.eye(len(POD)),
                                             atol=check_tol, rtol=0.):
            err = np.max(np.abs(POD.dot(POD) - np.eye(len(POD))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))
        elif product and not float_cmp_all(product.apply2(POD, POD), np.eye(len(POD)),
                                           atol=check_tol, rtol=0.):
            err = np.max(np.abs(product.apply2(POD, POD) - np.eye(len(POD))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))
        if len(POD) < len(EVECS):
            raise AccuracyError('additional orthonormalization removed basis vectors')

    return POD, SVALS
