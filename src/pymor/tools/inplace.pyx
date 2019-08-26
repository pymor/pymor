# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#cython: language_level=3

import cython
cimport numpy as np
from scipy.sparse import csr_matrix


@cython.boundscheck(False)
def iadd_masked(U, V, np.ndarray[np.int32_t, ndim=1] U_ind):
    assert len(U_ind) == len(V), 'Lengths of U_ind and V must match'
    assert U.shape[1:] == V.shape[1:], 'U.shape[1:] != V.shape[1:]'

    cdef unsigned int i, k, len_UU, dim
    cdef np.int32_t indi
    cdef np.ndarray[np.float64_t, ndim=2] UU
    cdef np.ndarray[np.float64_t, ndim=2] VV

    UU = U.reshape((len(U),-1))
    VV = V.reshape((len(V),-1))
    len_UU = len(UU)
    dim = UU.shape[1]

    for i in xrange(len(U_ind)):
        indi = U_ind[i]
        if indi < 0:
            continue
        if indi >= len_UU:
            raise IndexError('Index is too large!')
        for k in xrange(dim):
            UU[<unsigned int> indi, k] += VV[i, k]


@cython.boundscheck(False)
def isub_masked(U, V, np.ndarray[np.int32_t, ndim=1] U_ind):
    assert len(U_ind) == len(V), 'Lengths of U_ind and V must match'
    assert U.shape[1:] == V.shape[1:], 'U.shape[1:] != V.shape[1:]'

    cdef unsigned int i, k, len_UU, dim
    cdef np.int32_t indi
    cdef np.ndarray[np.float64_t, ndim=2] UU
    cdef np.ndarray[np.float64_t, ndim=2] VV

    UU = U.reshape((len(U),-1))
    VV = V.reshape((len(V),-1))
    len_UU = len(UU)
    dim = UU.shape[1]

    for i in xrange(len(U_ind)):
        indi = U_ind[i]
        if indi < 0:
            continue
        if indi >= len_UU:
            raise IndexError('Index is too large!')
        for k in xrange(dim):
            UU[<unsigned int> indi, k] -= VV[i, k]


@cython.boundscheck(False)
@cython.wraparound(False)
def set_unit_rows_cols(mat, np.ndarray[np.int32_t] indices):

    if not isinstance(mat, csr_matrix):
        raise NotImplementedError('Only csr_matrices are supported!')

    M, N = mat.shape
    if M != N:
        raise ValueError('matrix has to be square')

    cdef np.int32_t i, index, min_index, max_index
    min_index = 1
    max_index = -1
    for i in range(indices.shape[0]):
        index = indices[i]
        min_index = min(min_index, index)
        max_index = max(max_index, index)
    if min_index < 0:
        raise ValueError('indices have to be positive')
    if max_index > M:
        raise ValueError('indices to large')

    result = _set_unit_rows_cols_csr(mat.data, mat.indices, mat.indptr, M, N, indices)

    if result == -1:
        raise ValueError('matrix has to be square')
    elif result == -2:
        raise IndexError('missing diagonal entry in sparsity pattern')
    elif result != 0:
        raise RuntimeError('unknown problem occurred')


@cython.boundscheck(False)
@cython.wraparound(False)
def clear_rows_cols(mat, np.ndarray[np.int32_t] row_indices, np.ndarray[np.int32_t] col_indices):

    if not isinstance(mat, csr_matrix):
        raise NotImplementedError('Only csr_matrices are supported!')

    M, N = mat.shape
    cdef np.int32_t i, index, len_indices, min_index, max_index
    min_index = 1
    max_index = -1
    len_indices = row_indices.shape[0]
    for i in range(len_indices):
        index = row_indices[i]
        min_index = min(min_index, index)
        max_index = max(max_index, index)
    if min_index < 0:
        raise ValueError('row_indices have to be positive')
    if max_index > M:
        raise ValueError('row_indices to large')
    min_index = 1
    max_index = -1
    len_indices = col_indices.shape[0]
    for i in range(len_indices):
        index = col_indices[i]
        min_index = min(min_index, index)
        max_index = max(max_index, index)
    if min_index < 0:
        raise ValueError('col_indices have to be positive')
    if max_index > M:
        raise ValueError('col_indices to large')

    result = _clear_rows_cols_csr(mat.data, mat.indices, mat.indptr, M, N, row_indices, col_indices)

    if result != 0:
        raise RuntimeError('unknown problem occurred')


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _set_unit_rows_cols_csr(
        np.ndarray[np.double_t] data, np.ndarray[np.int32_t] indcs, np.ndarray[np.int32_t] indptr,
        np.int32_t M, np.int32_t N,
        np.ndarray[np.int32_t] indices):

    if M != N:
        return -1

    cdef np.int32_t i, j, m, n, index, len_indices
    cdef bint clear_row, clear_col, diagonal_found
    len_indices = indices.shape[0]

    for m in range(M):
        clear_row = False
        for i in range(len_indices): # not optimal, order n log(n), increment m and i simultaneously
            index = indices[i]
            if m == index:
                clear_row = True
                break
        if clear_row:
            diagonal_found = False
            for j in range(indptr[m], indptr[m + 1]):
                n = indcs[j]
                if n == m:
                    data[j] = 1.
                    diagonal_found = True
                else:
                    data[j] = 0.
            if not diagonal_found:
                return -2
        else: # we still need to clear all column entries in this row which are in indices
            for j in range(indptr[m], indptr[m + 1]):
                n = indcs[j]
                clear_col = False
                for i in range(len_indices): # not optimal, see above
                    index = indices[i]
                    if n == index:
                        clear_col = True
                        break
                if clear_col:
                    if n == m:
                        data[j] = 1.
                    else:
                        data[j] = 0.
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _clear_rows_cols_csr(
        np.ndarray[np.double_t] data, np.ndarray[np.int32_t] indices, np.ndarray[np.int32_t] indptr,
        np.int32_t M, np.int32_t N,
        np.ndarray[np.int32_t] row_indices,
        np.ndarray[np.int32_t] col_indices):

    cdef np.int32_t i, j, m, n, index, len_row_indices, len_col_indices
    cdef bint clear_row, clear_col
    len_row_indices = row_indices.shape[0]
    len_col_indices = col_indices.shape[0]

    if len_row_indices == 0 and len_row_indices == 0:
        return 0

    for m in range(M):
        clear_row = False
        for i in range(len_row_indices): # not optimal, order n log(n), increment m and i simultaneously
            index = row_indices[i]
            if m == index:
                clear_row = True
                break
        if clear_row:
            for j in range(indptr[m], indptr[m + 1]):
                n = indices[j]
                data[j] = 0.
        else: # we still need to clear all column entries in this row which are in col_indices
            for j in range(indptr[m], indptr[m + 1]):
                n = indices[j]
                clear_col = False
                for i in range(len_col_indices): # not optimal, see above
                    index = row_indices[i]
                    if n == index:
                        clear_col = True
                        break
                if clear_col:
                    data[j] = 0.
    return 0
