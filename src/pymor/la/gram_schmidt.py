# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
#
# Contributors: Andreas Buhr <andreas@andreasbuhr.de>

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.defaults import defaults
from pymor.core.exceptions import AccuracyError
from pymor.core.logger import getLogger
from pymor.tools.floatcmp import float_cmp_all


@defaults('atol', 'rtol', 'find_duplicates', 'reiterate', 'reiteration_threshold', 'check', 'check_tol')
def gram_schmidt(A, product=None, atol=1e-13, rtol=1e-13, offset=0, find_duplicates=True,
                 reiterate=True, reiteration_threshold=1e-1, check=True, check_tol=1e-3,
                 copy=False):
    """Orthonormalize a |VectorArray| using the stabilized Gram-Schmidt algorithm.

    Parameters
    ----------
    A
        The |VectorArray| which is to be orthonormalized.
    product
        The scalar product w.r.t. which to orthonormalize, given as a linear
        |Operator|. If `None` the Euclidean product is used.
    atol
        Vectors of norm smaller than `atol` are removed from the array.
    rtol
        Relative tolerance used to detect linear dependent vectors
        (which are then removed from the array).
    offset
        Assume that the first `offset` vectors are already orthogonal and start the
        algorithm at the `offset + 1`-th vector.
    find_duplicates
        If `True`, eliminate duplicate vectors before the main loop.
    reiterate
        If `True`, orthonormalize again if the norm of the orthogonalized vector is
        much smaller than the norm of the original vector.
    reiteration_threshold
        If `reiterate` is `True`, re-orthonormalize if the ratio between the norms of
        the orthogonalized vector and the original vector is smaller than this value.
    check
        If `True`, check if the resulting VectorArray is really orthonormal.
    check_tol
        Tolerance for the check.
    copy
        If `True`, create a copy of `A` instead of modifying `A` itself.


    Returns
    -------
    The orthonormalized |VectorArray|.
    """

    logger = getLogger('pymor.la.gram_schmidt.gram_schmidt')

    if copy:
        A = A.copy()

    # find duplicate vectors since in some circumstances these cannot be detected in the main loop
    # (is this really needed or is in this cases the tolerance poorly chosen anyhow)
    if find_duplicates:
        i = 0
        while i < len(A):
            duplicates = A.almost_equal(A, ind=i, o_ind=np.arange(max(offset, i + 1), len(A)))
            if np.any(duplicates):
                A.remove(np.where(duplicates)[0])
                logger.info("Removing duplicate vectors")
            i += 1

    # main loop
    remove = []
    norm = None
    for i in xrange(offset, len(A)):
        # first calculate norm
        if product is None:
            initial_norm = A.l2_norm(ind=i)[0]
        else:
            initial_norm = np.sqrt(product.apply2(A, A, V_ind=i, U_ind=i, pairwise=True))[0]

        if initial_norm < atol:
            logger.info("Removing vector {} of norm {}".format(i, initial_norm))
            remove.append(i)
            continue

        if i == 0:
            A.scal(1/initial_norm, ind=0)

        else:
            first_iteration = True

            # If reiterate is True, reiterate as long as the norm of the vector changes
            # strongly during orthonormalization (due to Andreas Buhr).
            while first_iteration or reiterate and norm < reiteration_threshold:

                if first_iteration:
                    first_iteration = False
                else:
                    logger.info('Orthonormalizing vector {} again'.format(i))

                # orthogonalize to all vectors left
                for j in xrange(i):
                    if j in remove:
                        continue
                    if product is None:
                        p = A.dot(A, ind=i, o_ind=j, pairwise=True)[0]
                    else:
                        p = product.apply2(A, A, V_ind=i, U_ind=j, pairwise=True)[0]
                    A.axpy(-p, A, ind=i, x_ind=j)

                # calculate new norm
                if product is None:
                    norm = A.l2_norm(ind=i)[0]
                else:
                    norm = np.sqrt(product.apply2(A, A, V_ind=i, U_ind=i, pairwise=True))[0]

                # remove vector if it got too small:
                if norm / initial_norm < rtol:
                    logger.info("Removing linear dependent vector {}".format(i))
                    remove.append(i)
                    break

                A.scal(1 / norm, ind=i)

    if remove:
        A.remove(remove)

    if check:
        orthogonal_indicator = A.dot(A, pairwise=False) if not product else product.apply2(A, A, pairwise=False)
        # do not check the vectors below the offset
        not_orthonormal_ones = []
        for ii in np.arange(offset):
            if not float_cmp_all(orthogonal_indicator[ii][ii], 1.0, check_tol):
                not_orthonormal_ones.append(ii)
                orthogonal_indicator[ii][ii] = 1.0
        # but at least warn about them
        if not_orthonormal_ones:
            logger.warn('vector{} {} {} not orthogonal (continuing anyway, since offset is {})'.format(
                's' if len(not_orthonormal_ones) > 1 else '',
                not_orthonormal_ones,
                'are' if len(not_orthonormal_ones) > 1 else 'is',
                offset))
        # then check (and fail) only for the new ones
        if not float_cmp_all(orthogonal_indicator, np.eye(len(A)), check_tol):
            err = np.max(np.abs(orthogonal_indicator - np.eye(len(A))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))


    return A
