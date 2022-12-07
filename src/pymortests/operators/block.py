# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla

from pymor.algorithms.projection import project
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test_apply():
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
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Aop.apply(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(A.dot(v), w)


def test_apply_adjoint():
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
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Aop.apply_adjoint(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(A.T.dot(v), w)


def test_block_diagonal():
    A = np.random.randn(2, 3)
    B = np.random.randn(4, 5)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))
    assert Cop.source.dim == 8
    assert Cop.range.dim == 6


def test_blk_diag_apply_inverse():
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Cop.apply_inverse(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(spla.solve(C, v), w)


def test_blk_diag_apply_inverse_adjoint():
    A = np.random.randn(2, 2)
    B = np.random.randn(3, 3)
    C = spla.block_diag(A, B)
    Aop = NumpyMatrixOperator(A)
    Bop = NumpyMatrixOperator(B)
    Cop = BlockDiagonalOperator((Aop, Bop))

    v1 = np.random.randn(2)
    v2 = np.random.randn(3)
    v = np.hstack((v1, v2))
    v1va = NumpyVectorSpace.from_numpy(v1)
    v2va = NumpyVectorSpace.from_numpy(v2)
    vva = BlockVectorSpace.make_array((v1va, v2va))

    wva = Cop.apply_inverse_adjoint(vva)
    w = np.hstack((wva.blocks[0].to_numpy(), wva.blocks[1].to_numpy()))
    assert np.allclose(spla.solve(C.T, v), w)


def test_sparse():
    np.random.seed(0)

    # construct an operator with parametric parts outside the diagonal
    ops = np.zeros((4, 4), dtype=object)
    for I in range(3):
        ops[I, I] = NumpyMatrixOperator(np.random.randn(3, 3))
    ops[3, 3] = NumpyMatrixOperator(np.random.randn(2, 2))
    ops[0, 2] = ProjectionParameterFunctional('mu', 1, 0) * NumpyMatrixOperator(np.random.randn(3, 3))

    # call operator
    block_op = BlockOperator(ops)
    assert block_op.parametric

    # assemble with a parameter
    mu = block_op.parameters.parse([1])
    assembled_op = block_op.assemble(mu=mu)
    assert assembled_op != block_op

    # construct another operator
    ops2 = ops.copy()
    ops2[1, 2] = ExpressionParameterFunctional('0*mu[0]', {'mu': 1}) * NumpyMatrixOperator(np.random.randn(3, 3))
    block_op_2 = BlockOperator(ops2)

    # densify
    block_op_2_dense = block_op_2.to_dense()
    dense_block_op = block_op.to_dense()
    sparse_block_op = dense_block_op.to_sparse()
    ones = block_op.range.ones()
    a = dense_block_op.apply2(ones, ones, mu)
    b = block_op.apply2(ones, ones, mu)
    c = sparse_block_op.apply2(ones, ones, mu)
    assert a == b == c

    # assembly
    assembled_sum = (0.5 * block_op + 0.5 * block_op_2).assemble(mu)
    assembled_sum_2 = (0.5 * block_op + 0.5 * block_op_2_dense).assemble(mu)
    a = block_op.apply2(ones, ones, mu)
    b = block_op_2.apply2(ones, ones, mu)
    c = assembled_sum.apply2(ones, ones, mu)
    d = assembled_sum_2.apply2(ones, ones, mu)
    # all have the same result
    assert a == b == c == d

    # other methods
    block_op.apply(ones, mu=mu)
    block_op.apply_adjoint(ones, mu=mu)
    block_op.d_mu('something')
    block_op.H

    # call as a list
    block_op_half = BlockOperator([[ops[0, 0], None], [None, ops[1, 1]]])
    assert not block_op_half.parametric
    # call with (data, coords) functionality
    block_op_ = BlockOperator((block_op.blocks.data, block_op.blocks.coords))
    assert block_op.apply2(ones, ones, mu=mu) == block_op_.apply2(ones, ones, mu=mu)

    # BlockDiagOperator
    block_diag_1 = BlockDiagonalOperator([ops[0, 0], ops[0, 2]])
    block_diag_2 = BlockDiagonalOperator([ops2[0, 0], ops2[1, 2]])
    (block_diag_1 + block_diag_2).assemble(mu)

    # projection
    projected_block = project(block_op, block_op.source.ones(), block_op.source.ones())
    projected_block_dense = project(dense_block_op, block_op.source.ones(), block_op.source.ones())
    projected_block.assemble(mu).matrix == projected_block_dense.assemble(mu).matrix
