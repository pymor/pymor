# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import cython
cimport numpy as np


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
