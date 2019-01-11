# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger


@defaults('atol', 'rtol', 'reiterate', 'reiteration_threshold', 'check', 'check_tol')
def gram_schmidt(A, product=None, atol=1e-13, rtol=1e-13, offset=0,
                 reiterate=True, reiteration_threshold=1e-1, check=True, check_tol=1e-3,
                 copy=True):
    """Orthonormalize a |VectorArray| using the stabilized Gram-Schmidt algorithm.

    Parameters
    ----------
    A
        The |VectorArray| which is to be orthonormalized.
    product
        The inner product |Operator| w.r.t. which to orthonormalize.
        If `None`, the Euclidean product is used.
    atol
        Vectors of norm smaller than `atol` are removed from the array.
    rtol
        Relative tolerance used to detect linear dependent vectors
        (which are then removed from the array).
    offset
        Assume that the first `offset` vectors are already orthonormal and start the
        algorithm at the `offset + 1`-th vector.
    reiterate
        If `True`, orthonormalize again if the norm of the orthogonalized vector is
        much smaller than the norm of the original vector.
    reiteration_threshold
        If `reiterate` is `True`, re-orthonormalize if the ratio between the norms of
        the orthogonalized vector and the original vector is smaller than this value.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` in-place.

    Returns
    -------
    The orthonormalized |VectorArray|.
    """

    logger = getLogger('pymor.algorithms.gram_schmidt.gram_schmidt')

    if copy:
        A = A.copy()

    # main loop
    remove = []
    for i in range(offset, len(A)):
        # first calculate norm
        initial_norm = A[i].norm(product)[0]

        if initial_norm < atol:
            logger.info("Removing vector {} of norm {}".format(i, initial_norm))
            remove.append(i)
            continue

        if i == 0:
            A[0].scal(1/initial_norm)

        else:
            first_iteration = True
            norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during orthonormalization (due to Andreas Buhr).
            while first_iteration or reiterate and norm/old_norm < reiteration_threshold:

                if first_iteration:
                    first_iteration = False
                else:
                    logger.info('Orthonormalizing vector {} again'.format(i))

                # orthogonalize to all vectors left
                for j in range(i):
                    if j in remove:
                        continue
                    p = A[i].pairwise_inner(A[j], product)[0]
                    A[i].axpy(-p, A[j])

                # calculate new norm
                old_norm, norm = norm, A[i].norm(product)[0]

                # remove vector if it got too small:
                if norm / initial_norm < rtol:
                    logger.info("Removing linear dependent vector {}".format(i))
                    remove.append(i)
                    break

            if norm > 0:
                A[i].scal(1 / norm)

    if remove:
        del A[remove]

    if check:
        error_matrix = A[offset:len(A)].inner(A, product)
        error_matrix[:len(A) - offset, offset:len(A)] -= np.eye(len(A) - offset)
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError('result not orthogonal (max err={})'.format(err))

    return A


def gram_schmidt_biorth(V, W, product=None, reiterate=True, reiteration_threshold=1e-1, check=True, check_tol=1e-3,
                        copy=True):
    """Biorthonormalize a pair of |VectorArrays| using the biorthonormal Gram-Schmidt process.

    See Algorithm 1 in [BKS11]_.

    Parameters
    ----------
    V, W
        The |VectorArrays| which are to be biorthonormalized.
    product
        The inner product |Operator| w.r.t. which to biorthonormalize.
        If `None`, the Euclidean product is used.
    reiterate
        If `True`, orthonormalize again if the norm of the orthogonalized vector is
        much smaller than the norm of the original vector.
    reiteration_threshold
        If `reiterate` is `True`, re-orthonormalize if the ratio between the norms of
        the orthogonalized vector and the original vector is smaller than this value.
    check
        If `True`, check if the resulting |VectorArray| is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `V` and `W` instead of modifying `V` and `W` in-place.


    Returns
    -------
    The biorthonormalized |VectorArrays|.
    """
    assert V.space == W.space
    assert len(V) == len(W)

    logger = getLogger('pymor.algorithms.gram_schmidt.gram_schmidt_biorth')

    if copy:
        V = V.copy()
        W = W.copy()

    # main loop
    for i in range(len(V)):
        # calculate norm of V[i]
        initial_norm = V[i].norm(product)[0]

        # project V[i]
        if i == 0:
            V[0].scal(1 / initial_norm)
        else:
            first_iteration = True
            norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during projection.
            while first_iteration or reiterate and norm / old_norm < reiteration_threshold:
                if first_iteration:
                    first_iteration = False
                else:
                    logger.info('Projecting vector V[{}] again'.format(i))

                for j in range(i):
                    # project by (I - V[j] * W[j]^T * E)
                    p = W[j].pairwise_inner(V[i], product)[0]
                    V[i].axpy(-p, V[j])

                # calculate new norm
                old_norm, norm = norm, V[i].norm(product)[0]

            if norm > 0:
                V[i].scal(1 / norm)

        # calculate norm of W[i]
        initial_norm = W[i].norm(product)[0]

        # project W[i]
        if i == 0:
            W[0].scal(1 / initial_norm)
        else:
            first_iteration = True
            norm = initial_norm
            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during projection.
            while first_iteration or reiterate and norm / old_norm < reiteration_threshold:
                if first_iteration:
                    first_iteration = False
                else:
                    logger.info('Projecting vector W[{}] again'.format(i))

                for j in range(i):
                    # project by (I - W[j] * V[j]^T * E)
                    p = V[j].pairwise_inner(W[i], product)[0]
                    W[i].axpy(-p, W[j])

                # calculate new norm
                old_norm, norm = norm, W[i].norm(product)[0]

            if norm > 0:
                W[i].scal(1 / norm)

        # rescale V[i]
        p = W[i].pairwise_inner(V[i], product)[0]
        V[i].scal(1 / p)

    if check:
        error_matrix = W.inner(V, product)
        error_matrix -= np.eye(len(V))
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError('Result not biorthogonal (max err={})'.format(err))

    return V, W
