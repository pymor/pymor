from __future__ import absolute_import, division, print_function

import numpy as np


def inv_two_by_two(A):
    '''retval[i1,...,ik,m,n] = numpy.linalg.inv(A[i1,...,ik,:,:])'''

    assert A.shape[-1] == A.shape[-2] == 2, ValueError('Wrong shape of argmument.')

    D = A[..., 0, 0] * A[..., 1, 1] - A[..., 1, 0] * A[..., 0, 1]
    D = 1 / D

    INV = np.empty_like(A)
    INV[..., 0, 0] = A[..., 1, 1]
    INV[..., 1, 1] = A[..., 0, 0]
    INV[..., 1, 0] = - A[..., 1, 0]
    INV[..., 0, 1] = - A[..., 0, 1]
    INV *= D[..., np.newaxis, np.newaxis]

    return INV


def inv_transposed_two_by_two(A):
    '''retval[i1,...,ik,m,n] = numpy.linalg.inv(A[i1,...,ik,:,:])'''

    assert A.shape[-1] == A.shape[-2] == 2, ValueError('Wrong shape of argmument.')

    D = A[..., 0, 0] * A[..., 1, 1] - A[..., 1, 0] * A[..., 0, 1]
    D = 1 / D

    INV = np.empty_like(A)
    INV[..., 0, 0] = A[..., 1, 1]
    INV[..., 1, 1] = A[..., 0, 0]
    INV[..., 1, 0] = - A[..., 0, 1]
    INV[..., 0, 1] = - A[..., 1, 0]
    INV *= D[..., np.newaxis, np.newaxis]

    return INV
