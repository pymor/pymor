# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse import issparse

from pymor.core import defaults
from pymor.la.numpyvectorarray import NumpyVectorArray
from pymor.operators.numpy import NumpyLinearOperator


def solve_linear_numpy_bicgstab(A, U, ind=None, tol=None, maxiter=None):
    assert isinstance(A, NumpyLinearOperator)
    assert isinstance(U, NumpyVectorArray)
    assert A.dim_range == U.dim

    tol =  defaults.bicgstab_tol if tol is None else tol
    maxiter = defaults.bicgstab_maxiter if maxiter is None else maxiter

    A = A._matrix
    U = U._array if ind is None else U._array[ind]
    if U.shape[1] == 0:
        return NumpyVectorArray(U)
    R = np.empty((len(U), A.shape[1]))
    if issparse(A):
        for i, UU in enumerate(U):
            R[i], _ = bicgstab(A, UU, tol=tol, maxiter=maxiter)
    else:
        for i, UU in enumerate(U):
            R[i] = np.linalg.solve(A, UU)
    return NumpyVectorArray(R)


def solve_linear(A, U, ind=None, mu=None, **kwargs):
    assert A.dim_range == U.dim
    if not A.assembled:
        A = A.assemble(mu)
    else:
        assert mu is None
    if isinstance(A, NumpyLinearOperator):
        return solve_linear_numpy_bicgstab(A, U, ind, **kwargs)
    else:
        raise NotImplementedError
