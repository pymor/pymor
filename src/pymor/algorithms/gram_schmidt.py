# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger


@defaults('atol', 'rtol', 'reiterate', 'reiteration_threshold', 'check', 'check_tol')
def gram_schmidt(A, product=None, atol=1e-13, rtol=1e-13, offset=0, find_duplicates=True,
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
    find_duplicates
        unused


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
        if product is None:
            initial_norm = A[i].l2_norm()[0]
        else:
            initial_norm = np.sqrt(product.pairwise_apply2(A[i], A[i]))[0]

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
                    if product is None:
                        p = A[i].pairwise_dot(A[j])[0]
                    else:
                        p = product.pairwise_apply2(A[i], A[j])[0]
                    A[i].axpy(-p, A[j])

                # calculate new norm
                if product is None:
                    old_norm, norm = norm, A[i].l2_norm()[0]
                else:
                    old_norm, norm = norm, np.sqrt(product.pairwise_apply2(A[i], A[i])[0])

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
        if product:
            error_matrix = product.apply2(A[offset:len(A)], A)
        else:
            error_matrix = A[offset:len(A)].dot(A)
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

    .. [BKS11]  P. Benner, M. Köhler, J. Saak,
                Sparse-Dense Sylvester Equations in :math:`\mathcal{H}_2`-Model Order Reduction,
                Max Planck Institute Magdeburg Preprint, available from http://www.mpi-magdeburg.mpg.de/preprints/,
                2011.

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
        if product is None:
            initial_norm = V[i].l2_norm()[0]
        else:
            initial_norm = np.sqrt(product.pairwise_apply2(V[i], V[i]))[0]

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
                    if product is None:
                        p = W[j].pairwise_dot(V[i])[0]
                    else:
                        p = product.pairwise_apply2(W[j], V[i])[0]
                    V[i].axpy(-p, V[j])

                # calculate new norm
                if product is None:
                    old_norm, norm = norm, V[i].l2_norm()[0]
                else:
                    old_norm, norm = norm, np.sqrt(product.pairwise_apply2(V[i], V[i])[0])

            if norm > 0:
                V[i].scal(1 / norm)

        # calculate norm of W[i]
        if product is None:
            initial_norm = W[i].l2_norm()[0]
        else:
            initial_norm = np.sqrt(product.pairwise_apply2(W[i], W[i]))[0]

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
                    if product is None:
                        p = V[j].pairwise_dot(W[i])[0]
                    else:
                        p = product.pairwise_apply2(V[j], W[i])[0]
                    W[i].axpy(-p, W[j])

                # calculate new norm
                if product is None:
                    old_norm, norm = norm, W[i].l2_norm()[0]
                else:
                    old_norm, norm = norm, np.sqrt(product.pairwise_apply2(W[i], W[i])[0])

            if norm > 0:
                W[i].scal(1 / norm)

        # rescale V[i]
        if product is None:
            p = W[i].pairwise_dot(V[i])[0]
        else:
            p = product.pairwise_apply2(W[i], V[i])[0]
        V[i].scal(1 / p)

    if check:
        if product:
            error_matrix = product.apply2(W, V)
        else:
            error_matrix = W.dot(V)
        error_matrix -= np.eye(len(V))
        if error_matrix.size > 0:
            err = np.max(np.abs(error_matrix))
            if err >= check_tol:
                raise AccuracyError('Result not biorthogonal (max err={})'.format(err))

    return V, W
