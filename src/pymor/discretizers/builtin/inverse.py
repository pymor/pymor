# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np


def inv_two_by_two(A):
    """Efficiently compute the inverses of a |NumPy array| of 2x2-matrices

    This implements ::

    |  retval[i1,...,ik,m,n] = numpy.linalg.inv(A[i1,...,ik,:,:]).

    """
    assert A.shape[-1] == A.shape[-2] == 2, 'Wrong shape of argument.'

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
    """Efficiently compute the tranposed inverses of a |Numpy array| of 2x2-matrices

    This implements ::

    |  retval[i1,...,ik,m,n] = numpy.linalg.inv(A[i1,...,ik,:,:]).

    """
    assert A.shape[-1] == A.shape[-2] == 2, 'Wrong shape of argument.'

    D = A[..., 0, 0] * A[..., 1, 1] - A[..., 1, 0] * A[..., 0, 1]
    D = 1 / D

    INV = np.empty_like(A)
    INV[..., 0, 0] = A[..., 1, 1]
    INV[..., 1, 1] = A[..., 0, 0]
    INV[..., 1, 0] = - A[..., 0, 1]
    INV[..., 0, 1] = - A[..., 1, 0]
    INV *= D[..., np.newaxis, np.newaxis]

    return INV
