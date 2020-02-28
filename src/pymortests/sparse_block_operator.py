import numpy as np
import scipy as sp
import pytest
from pymortests.base import runmodule

from pymor.algorithms.to_matrix import to_matrix
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.operators.block import SparseBlockOperator
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.floatcmp import float_cmp_all
from pymor.vectorarrays.block import BlockVectorArray, BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

ops = [{2: NumpyMatrixOperator(3*np.ones((1, 3)))},
       {0: NumpyMatrixOperator(4*np.ones((2, 1))), 2: NumpyMatrixOperator(6*np.ones((2, 3)))},
       {1: NumpyMatrixOperator(8*np.ones((3, 2)))}]
ops_mat = np.array(
        [[0, 0, 0, 3, 3, 3],
         [4, 0, 0, 6, 6, 6],
         [4, 0, 0, 6, 6, 6],
         [0, 8, 8, 0, 0, 0],
         [0, 8, 8, 0, 0, 0],
         [0, 8, 8, 0, 0, 0]])

identity_ops = [{0: NumpyMatrixOperator(np.eye(1))},
                {1: NumpyMatrixOperator(np.eye(2))},
                {2: NumpyMatrixOperator(np.eye(3))}]
identity_ops_mat = np.eye(6)

invertible_ops = [{1: NumpyMatrixOperator(np.array([0, 1]).reshape((1, 2)))},
                  {0: NumpyMatrixOperator(np.array([1, 0]).reshape((2, 1))),
                   1: NumpyMatrixOperator(np.array([[0, 0], [1, 0]]))},
                  {2: NumpyMatrixOperator(np.eye(3))}]
invertible_ops_mat = np.array(
        [[0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]])

theta = ProjectionParameterFunctional('param', ())
parametric_ops = [{0: identity_ops[0][0]},
                  {1: LincombOperator([identity_ops[1][1], identity_ops[1][1]], [theta, theta])},
                  {2: LincombOperator([identity_ops[2][2], identity_ops[2][2], identity_ops[2][2]],
                                      [theta, theta, theta])}]
def parametric_ops_mat(p):
    return np.array(
            [[1,   0,   0,   0,   0,   0],
             [0, 2*p,   0,   0,   0,   0],
             [0,   0, 2*p,   0,   0,   0],
             [0,   0,   0, 3*p,   0,   0],
             [0,   0,   0,   0, 3*p,   0],
             [0,   0,   0,   0,   0, 3*p]])
            

U = BlockVectorSpace([NumpyVectorSpace(1), NumpyVectorSpace(2), NumpyVectorSpace(3)]).ones(1)
V = BlockVectorSpace([NumpyVectorSpace(1), NumpyVectorSpace(2), NumpyVectorSpace(3)]).ones(1)


def test_init():
    SparseBlockOperator(ops)
    SparseBlockOperator(identity_ops)
    SparseBlockOperator(invertible_ops)
    SparseBlockOperator(parametric_ops)


def test_source_range():
    for blocks in (ops, identity_ops, invertible_ops, parametric_ops):
        op = SparseBlockOperator(ops)
        assert op.num_source_blocks == 3
        assert op.num_range_blocks == 3
        assert op.source == BlockVectorSpace([NumpyVectorSpace(1), NumpyVectorSpace(2), NumpyVectorSpace(3)])
        assert op.range == BlockVectorSpace([NumpyVectorSpace(1), NumpyVectorSpace(2), NumpyVectorSpace(3)])


def test_paramter_type():
    for blocks in (ops, identity_ops, invertible_ops):
        op = SparseBlockOperator(ops)
        assert not op.parametric
    op = SparseBlockOperator(parametric_ops)
    assert op.parameter_type == theta.parameter_type


def test_H():
    op = SparseBlockOperator(ops)
    ad_op = op.H
    assert ad_op.num_source_blocks == op.num_range_blocks
    assert ad_op.num_range_blocks == op.num_source_blocks

    def is_same(op1, op2):
        return op1.source == op2.source and op1.range == op2.range and np.allclose(op1.matrix, op2.matrix)

    assert set(ad_op.blocks[0].keys()) == set([1,])
    assert is_same(ad_op.blocks[0][1], op.blocks[1][0].H)
    assert set(ad_op.blocks[1].keys()) == set([2,])
    assert is_same(ad_op.blocks[1][2], op.blocks[2][1].H)
    assert set(ad_op.blocks[2].keys()) == set([0, 1])
    assert is_same(ad_op.blocks[2][0], op.blocks[0][2].H)
    assert is_same(ad_op.blocks[2][1], op.blocks[1][2].H)


def test_to_matrix():
    for blocks, expected_mat in ((ops, ops_mat),
                                 (identity_ops, identity_ops_mat),
                                 (invertible_ops, invertible_ops_mat)):
        op = SparseBlockOperator(blocks)
        mat = to_matrix(op, format='dense')
        assert float_cmp_all(mat, expected_mat)
    parametric_op = SparseBlockOperator(parametric_ops)
    for mu in (-1.2, 0, 3):
        mat = to_matrix(parametric_op, format='dense', mu=mu)
        assert float_cmp_all(mat, parametric_ops_mat(mu))


def test_assemble():
    for blocks in (ops, identity_ops, invertible_ops):
        op = SparseBlockOperator(blocks)
        assembled_op = op.assemble()
        assert float_cmp_all(to_matrix(assembled_op, format='dense'), to_matrix(op, format='dense'))
    op = SparseBlockOperator(parametric_ops)
    for mu in (-1.2, 0, 3):
        assembled_op = op.assemble(mu=mu)
        assert float_cmp_all(to_matrix(assembled_op, format='dense'), parametric_ops_mat(mu))


def test_apply():
    for blocks in (ops, identity_ops, invertible_ops):
        op = SparseBlockOperator(blocks)
        V = op.apply(U)
        assert float_cmp_all(V.to_numpy(),
                             to_matrix(op, format='dense').dot(U.to_numpy().T).T)
    for mu in (-1.2, 0, 3):
        op = SparseBlockOperator(parametric_ops)
        V = op.apply(U, mu=mu)
        assert float_cmp_all(V.to_numpy(),
                             to_matrix(op, format='dense', mu=mu).dot(U.to_numpy().T).T)


def test_apply_adjoint():
    for blocks in (ops, identity_ops, invertible_ops):
        op = SparseBlockOperator(blocks)
        U = op.apply_adjoint(V)
        assert float_cmp_all(U.to_numpy(),
                             to_matrix(op, format='dense').T.dot(V.to_numpy().T).T)
    for mu in (-1.2, 0, 3):
        op = SparseBlockOperator(parametric_ops)
        U = op.apply_adjoint(V, mu=mu)
        assert float_cmp_all(U.to_numpy(),
                             to_matrix(op, format='dense', mu=mu).T.dot(V.to_numpy().T).T)


def test_apply_inverse():
    op = SparseBlockOperator(ops)
    U = op.apply_inverse(V)
    expected_U, _ = sp.sparse.linalg.lgmres(to_matrix(op), V.to_numpy().ravel())
    assert float_cmp_all(U.to_numpy(), expected_U)
    for blocks in (identity_ops, invertible_ops):
        op = SparseBlockOperator(blocks)
        U = op.apply_inverse(V)
        assert float_cmp_all(U.to_numpy(),
                             np.linalg.solve(to_matrix(op, format='dense'), V.to_numpy().ravel()))
    for mu in (-1.2, 3):
        op = SparseBlockOperator(parametric_ops)
        U = op.apply_inverse(V, mu=mu)
        assert float_cmp_all(U.to_numpy(),
                             np.linalg.solve(to_matrix(op, format='dense', mu=mu), V.to_numpy().ravel()))


def test_assemble_lincomb_op():
    for blocks in (ops, identity_ops, invertible_ops):
        assert float_cmp_all(to_matrix(SparseBlockOperator(blocks) + 2*SparseBlockOperator(blocks), format='dense'),
                             to_matrix(SparseBlockOperator(blocks), format='dense') +
                             2*to_matrix(SparseBlockOperator(blocks), format='dense'))


def test_as_range_array():
    for blocks in (ops, identity_ops, invertible_ops):
        op = SparseBlockOperator(blocks)
        V = op.as_range_array()
        assert float_cmp_all(to_matrix(op, format='dense').T, V.to_numpy())
    op = SparseBlockOperator(parametric_ops)
    for mu in (-1.2, 0, 3):
        V = op.as_range_array(mu=mu)
        assert float_cmp_all(to_matrix(op, format='dense', mu=mu).T, V.to_numpy())


def test_as_source_array():
    for blocks in (ops, identity_ops, invertible_ops):
        op = SparseBlockOperator(blocks)
        U = op.as_source_array()
        assert float_cmp_all(to_matrix(op, format='dense'), U.to_numpy())
    op = SparseBlockOperator(parametric_ops)
    for mu in (-1.2, 0, 3):
        U = op.as_range_array(mu=mu)
        assert float_cmp_all(to_matrix(op, format='dense', mu=mu), U.to_numpy())

if __name__ == "__main__":
    runmodule(filename=__file__)
