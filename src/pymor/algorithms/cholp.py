# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import print_function, division, absolute_import

import numpy as np


def cholp(A, copy=True):
    """Low-rank approximation using pivoted Cholesky decomposition

    .. note::

        Should be replaced with LAPACK routine DPSTRF (when it becomes available in NumPy).

    .. [H02] N. J. Higham, Accuracy and Stability of Numerical Algorithms,
             Second edition, Society for Industrial and Applied Mathematics,
             Philadelphia, PA, 2002; sec. 10.3.

    Parameters
    ----------
    A
        Symmetric positive semidefinite matrix as |NumPy array|.
    copy
        Should A be copied.
    """
    assert isinstance(A, np.ndarray)
    assert A.shape[0] == A.shape[1]

    if copy:
        A = A.copy()

    n = A.shape[0]
    piv = np.arange(n)
    I = 0

    for i in range(n):
        d = A.diagonal()
        j = np.argmax(d[i:])
        j += i
        a_max = d[j]
        if i == 0:
            a_max_1 = a_max
        elif a_max <= 0.5 * n * np.finfo(float).eps * a_max_1:
            I = i
            break

        # Symmetric row/column permutation.
        if j != i:
            A[:, [i, j]] = A[:, [j, i]]
            A[[i, j], :] = A[[j, i], :]
            piv[[i, j]] = piv[[j, i]]

        A[i, i] = np.sqrt(A[i, i])
        if i == n:
            break
        A[i + 1:, i] /= A[i, i]

        # For simplicity update the whole of the remaining submatrix (rather
        # than just the upper triangle).
        A[i + 1:, i + 1:] -= np.outer(A[i + 1:, i], A[i + 1:, i])

    L = np.tril(A)
    if I > 0:
        L = L[:, :I]
    ipiv = np.arange(n)
    for i, j in enumerate(piv):
        ipiv[j] = i
    L = L[ipiv, :]

    return L
