# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import division

import numpy as np
import cython
cimport numpy as np

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t


@cython.boundscheck(False)
def inverse_relation(np.ndarray[DTYPE_t, ndim=2] R, size_rhs=None, with_indices=False, unsafe=False):
    assert R.ndim == 2
    cdef int i
    cdef int j
    cdef int x
    cdef np.ndarray[DTYPE_t, ndim=2] RINV
    cdef np.ndarray[DTYPE_t, ndim=2] RINVI
    cdef np.ndarray[DTYPE_t, ndim=1] RINV_COL_COUNTS

    num_columns_RINV = np.bincount(R.ravel()).max()
    if size_rhs is None:
        size_rhs = R.max() + 1
    elif not unsafe:
        assert size_rhs >= R.max() + 1

    RINV = np.empty((size_rhs, num_columns_RINV), dtype=DTYPE)
    RINV.fill(-1)
    if with_indices:
        RINVI = np.empty_like(RINV)
        RINVI.fill(-1)

    RINV_COL_COUNTS = np.zeros(size_rhs, dtype=DTYPE)

    if not with_indices:
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                x = R[<unsigned int>i, <unsigned int>j]
                if x >= 0:
                    RINV[<unsigned int>x, <unsigned int>RINV_COL_COUNTS[<unsigned int>x]] = i
                    RINV_COL_COUNTS[<unsigned int>x] += 1
        return RINV, RINVI
    else:
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                x = R[<unsigned int>i, <unsigned int>j]
                if x >= 0:
                    RINV[<unsigned int>x, <unsigned int>RINV_COL_COUNTS[<unsigned int>x]] = i
                    RINVI[<unsigned int>x, RINV_COL_COUNTS[<unsigned int>x]] = j
                    RINV_COL_COUNTS[<unsigned int>x] += 1
        return RINV, RINVI
