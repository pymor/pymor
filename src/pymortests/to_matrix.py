# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps

from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.block import BlockDiagonalOperator
from pymor.operators.constructions import (AdjointOperator, Concatenation, IdentityOperator, LincombOperator,
                                           VectorArrayOperator)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace


def test_to_matrix():
    np.random.seed(0)
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = np.random.randn(3, 3)

    X = np.bmat([[np.eye(2) + A, np.zeros((2, 3))], [np.zeros((3, 2)), B.dot(C.T)]])

    C = sps.csc_matrix(C)

    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = NumpyMatrixOperator(C)

    Xop = BlockDiagonalOperator([LincombOperator([IdentityOperator(NumpyVectorSpace(2)), Aop],
                                                 [1, 1]), Concatenation(Bop, AdjointOperator(Cop))])

    assert np.allclose(X, to_matrix(Xop))
    assert np.allclose(X, to_matrix(Xop, format='csr').toarray())

    np.random.seed(0)
    V = np.random.randn(10, 2)
    Vva = NumpyVectorArray(V.T)
    Vop = VectorArrayOperator(Vva)
    assert np.allclose(V, to_matrix(Vop))
    Vop = VectorArrayOperator(Vva, transposed=True)
    assert np.allclose(V, to_matrix(Vop).T)
