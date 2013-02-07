from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import defaults
from pymor.core.exceptions import AccuracyError
from pymor.la import float_cmp_all
from pymor.discreteoperators import DiscreteOperatorInterface, GenericLinearOperator


def gram_schmidt(A, product=None, tol=None, row_offset=0, find_row_duplicates=True, find_col_duplicates=False,
                 check=None, check_tol=None):

    A = A.copy()
    if product is not None and not isinstance(product, DiscreteOperatorInterface):
        product = GenericLinearOperator(product)
    tol = defaults.gram_schmidt_tol if tol is None else tol
    check = defaults.gram_schmidt_tol if check is None else check
    check_tol = check_tol or defaults.gram_schmidt_check_tol

    # find duplicate rows since in some circumstances these cannot be detected in the main loop
    # (is this really needed or is in this cases the tolerance poorly chosen anyhow)
    if find_row_duplicates:
        for i in xrange(A.shape[0]):
            for j in xrange(max(row_offset, i + 1), A.shape[0]):
                if float_cmp_all(A[i], A[j]):
                    A[j] = 0

    # find duplicate columns
    if find_col_duplicates:
        for i in xrange(A.shape[1]):
            for j in xrange(i, A.shape[1]):
                if float_cmp_all(A[:, i], A[:, j]):
                    A[:, j] = 0

    # main loop
    for i in xrange(A.shape[0]):

        if i >= row_offset:
            if product is None:
                norm = np.sqrt(np.sum(A[i] ** 2))
            else:
                norm = np.sqrt(product.apply2(A[i], A[i]))

            if norm < tol:
                A[i] = 0
            else:
                A[i] = A[i] / norm

        j = max(row_offset, i + 1)
        if product is None:
            p = np.sum(A[j:] * A[i], axis=-1)
        else:
            p = product.apply2(A[j:], A[i], pairwise=False)

        A[j:] -= p[..., np.newaxis] * A[i]

    rows = np.logical_not(np.all(A == 0, axis=-1))
    A = A[rows]

    if check:
        if not float_cmp_all(A.dot(A.T), np.eye(A.shape[0]), check_tol):
            err = np.max(np.abs(A.dot(A.T) - np.eye(A.shape[0])))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))

    return A
