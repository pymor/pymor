# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import defaults
from pymor.core.exceptions import AccuracyError
from pymor.tools import float_cmp_all
from pymor.operators import OperatorInterface


def gram_schmidt(A, product=None, tol=None, offset=0, find_duplicates=True,
                 check=None, check_tol=None):
    '''Orthonormnalize a matrix using the Gram-Schmidt algorithm.

    Parameters
    ----------
    A
        The VectorArray which is to be orthonormalized.
    product
        The scalar product w.r.t. which to orthonormalize.
    tol
        Tolerance to determine a linear dependent row.
    offset
        Assume that the first `offset` vectors are already orthogonal and start the
        algorithm at the `offset + 1`-th vector.
    find_duplicates
        If `True`, eliminate duplicate vectors before the main loop.
    check
        If `True`, check if the resulting VectorArray is really orthonormal. If `None`, use
        `defaults.gram_schmidt_check`.
    check_tol
        Tolerance for the check. If `None`, `defaults.gram_schmidt_check_tol` is used.


    Returns
    -------
    The orthonormalized matrix.
    '''

    tol = defaults.gram_schmidt_tol if tol is None else tol
    check = defaults.gram_schmidt_tol if check is None else check
    check_tol = check_tol or defaults.gram_schmidt_check_tol

    # find duplicate vectors since in some circumstances these cannot be detected in the main loop
    # (is this really needed or is in this cases the tolerance poorly chosen anyhow)
    if find_duplicates:
        for i in xrange(len(A)):
            duplicates = A.almost_equal(A, ind=[i], o_ind=xrange(max(offset, i + 1), len(A)))
            if np.any(duplicates):
                A.remove(np.where(duplicates))

    # main loop
    i = 0
    remove = []
    while i < len(A):

        if i >= offset:
            if product is None:
                norm = A.l2_norm(ind=[i])[0]
            else:
                norm = np.sqrt(product.apply2(A, A, V_ind=[i], U_ind=[i], pairwise=True))[0]

            if norm < tol:
                remove.append(i)
                i += 1
                continue
            else:
                A.iadd_mult(None, factor=1/norm, o_factor=0, ind=[i])

        for j in xrange(max(offset, i + 1), len(A)):
            if product is None:
                p = A.prod(A, ind=[j], o_ind=[i], pairwise=True)[0]
            else:
                p = product.apply2(A, A, V_ind=[j], U_ind=[i], pairwise=True)[0]
            A.iadd_mult(A, o_factor=-p, ind=[j], o_ind=[i])

        i += 1

    if remove:
        A.remove(remove)

    if check:
        if not product and not float_cmp_all(A.prod(A, pairwise=False), np.eye(len(A)), check_tol):
            err = np.max(np.abs(A.prod(A, pairwise=False) - np.eye(len(A))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))
        elif product and not float_cmp_all(product.apply2(A, A, pairwise=False), np.eye(len(A)), check_tol):
            err = np.max(np.abs(product.apply2(A, A, pairwise=False) - np.eye(len(A))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))

    return A


def numpy_gram_schmidt(A, product=None, tol=None, row_offset=0, find_row_duplicates=True, find_col_duplicates=False,
                       check=None, check_tol=None):
    '''Orthonormnalize a matrix using the Gram-Schmidt algorithm.

    Parameters
    ----------
    A
        The matrix which is to be orthonormalized.
    product
        The scalar product w.r.t. which to orthonormalize. Either a `DiscreteLinearOperator`
        or a square matrix.
    tol
        Tolerance to determine a linear dependent row.
    row_offset
        Assume that the first `row_offset` rows are already orthogonal and start the
        algorithm at the `row_offset + 1`-th row.
    find_row_duplicates
        If `True`, eliminate duplicate rows before the main loop.
    find_col_duplicates
        If `True`, eliminate duplicate columns before the main loop.
    check
        If `True`, check if the resulting matrix is really orthonormal. If `None`, use
        `defaults.gram_schmidt_check`.
    check_tol
        Tolerance for the check. If `None`, `defaults.gram_schmidt_check_tol` is used.


    Returns
    -------
    The orthonormalized matrix.
    '''

    A = A.copy()
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
        if not product and not float_cmp_all(A.dot(A.T), np.eye(A.shape[0]), check_tol):
            err = np.max(np.abs(A.dot(A.T) - np.eye(A.shape[0])))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))
        elif product and not float_cmp_all(product.apply2(A, A, pairwise=False), np.eye(len(A)), check_tol):
            err = np.max(np.abs(product.apply2(A, A, pairwise=False) - np.eye(len(A))))
            raise AccuracyError('result not orthogonal (max err={})'.format(err))

    return A
