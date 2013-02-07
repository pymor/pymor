from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import defaults
from pymor.la import float_cmp_all
from pymor.discreteoperators import DiscreteOperatorInterface, GenericLinearOperator


def gram_schmidt(A, product=None, tol=None, find_row_duplicates=True, find_col_duplicates=False):

    A = A.copy()
    if product is not None and not isinstance(product, DiscreteOperatorInterface):
        product = GenericLinearOperator(product)
    tol = defaults.gram_schmidt_tol if tol is None else tol

    # find duplicate rows since in some circumstances these cannot be detected in the main loop
    # (is this really needed or is in this cases the tolerance poorly chosen anyhow)
    if find_row_duplicates:
        for i in xrange(A.shape[0]):
            for j in xrange(i + 1, A.shape[0]):
                if float_cmp_all(A[i], A[j]):
                    A[j] = 0

    # find duplicate columns
    if find_col_duplicates:
        for i in xrange(A.shape[1]):
            for j in xrange(i, A.shape[1]):
                if float_cmp_all(A[:, i], A[:, j]):
                    A[:, j] = 0

    # main loop
    if product is None:
        for i in xrange(A.shape[0]):
            norm = np.sqrt(np.sum(A[i] ** 2))

            if norm < tol:
                A[i] = 0
            else:
                A[i] = A[i] / norm

            p = np.sum(A[i + 1:] * A[i], axis=-1)
            A[i + 1:] -= p[..., np.newaxis] * A[i]
    else:
        for i in xrange(A.shape[0]):
            norm = np.sqrt(product.apply2(A[i], A[i]))

            if norm < tol:
                A[i] = 0
            else:
                A[i] = A[i] / norm

            p = product.apply2(A[i + 1:], A[i], pairwise=False)
            A[i + 1:] -= p[..., np.newaxis] * A[i]

    rows = np.logical_not(np.all(A == 0, axis=-1))
    return A[rows]
