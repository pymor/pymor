# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.numpy import NumpyVectorArray

def test_hstack():
    np.random.seed(0)
    A = np.random.randn(2, 3)
    B = np.random.randn(2, 4)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockOperator.hstack((Aop, Bop))
    assert Cop.source.dim == 7
    assert Cop.range.dim == 2

def test_vstack():
    np.random.seed(0)
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 3)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockOperator.vstack((Aop, Bop))
    assert Cop.source.dim == 3
    assert Cop.range.dim == 6

def test_apply():
    np.random.seed(0)

    A11 = np.random.randn(2, 3)
    A12 = np.random.randn(2, 4)
    A21 = np.zeros((5, 3))
    A22 = np.random.randn(5, 4)
    A = np.vstack((np.hstack((A11, A12)),
                   np.hstack((A21, A22))))
    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A22op = NumpyMatrixOperator(A22)
    Aop = BlockOperator(np.array([[A11op, A12op], [None, A22op]]))

    v1 = np.random.randn(3)
    v2 = np.random.randn(4)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorArray(v1)
    v2va = NumpyVectorArray(v2)
    vva = BlockVectorArray((v1va, v2va))

    wva = Aop.apply(vva)
    w = np.hstack((wva.block(0).data, wva.block(1).data))
    assert np.allclose(A.dot(v), w)

def test_apply_adjoint():
    np.random.seed(0)

    A11 = np.random.randn(2, 3)
    A12 = np.random.randn(2, 4)
    A21 = np.zeros((5, 3))
    A22 = np.random.randn(5, 4)
    A = np.vstack((np.hstack((A11, A12)),
                   np.hstack((A21, A22))))
    A11op = NumpyMatrixOperator(A11)
    A12op = NumpyMatrixOperator(A12)
    A22op = NumpyMatrixOperator(A22)
    Aop = BlockOperator(np.array([[A11op, A12op], [None, A22op]]))

    v1 = np.random.randn(2)
    v2 = np.random.randn(5)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorArray(v1)
    v2va = NumpyVectorArray(v2)
    vva = BlockVectorArray((v1va, v2va))

    wva = Aop.apply_adjoint(vva)
    w = np.hstack((wva.block(0).data, wva.block(1).data))
    assert np.allclose(A.T.dot(v), w)

def test_block_diagonal():
    np.random.seed(0)
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 5)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))
    assert Cop.source.dim == 8
    assert Cop.range.dim == 6

def test_blk_diag_apply_inverse():
    np.random.seed(0)

    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorArray(v1)
    v2va = NumpyVectorArray(v2)
    vva = BlockVectorArray((v1va, v2va))

    wva = Cop.apply_inverse(vva)
    w = np.hstack((wva.block(0).data, wva.block(1).data))
    assert np.allclose(spla.solve(C, v), w)

def test_blk_diag_apply_inverse_adjoint():
    np.random.seed(0)

    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorArray(v1)
    v2va = NumpyVectorArray(v2)
    vva = BlockVectorArray((v1va, v2va))

    wva = Cop.apply_inverse_adjoint(vva)
    w = np.hstack((wva.block(0).data, wva.block(1).data))
    assert np.allclose(spla.solve(C.T, v), w)
