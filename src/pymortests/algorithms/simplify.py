# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import numpy as np
import pytest

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.simplify import contract, expand
from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.block import BlockOperator
from pymor.operators.constructions import AdjointOperator, ConcatenationOperator, LincombOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace

pytestmark = pytest.mark.builtin


def test_expand():
    ops = [NumpyMatrixOperator(np.eye(1) * i) for i in range(8)]
    pfs = [ProjectionParameterFunctional('p', 9, i) for i in range(8)]
    prods = [o * p for o, p in zip(ops, pfs, strict=True)]

    op = ((prods[0] + prods[1] + prods[2]) @ (prods[3] + prods[4] + prods[5]) @
          (prods[6] + prods[7]))

    eop = expand(op)

    assert isinstance(eop, LincombOperator)
    assert len(eop.operators) == 3 * 3 * 2
    assert all(isinstance(o, ConcatenationOperator) and len(o.operators) == 3
               for o in eop.operators)
    assert ({to_matrix(o)[0, 0] for o in eop.operators}
            == {i0 * i1 * i2 for i0, i1, i2 in product([0, 1, 2], [3, 4, 5], [6, 7])})
    assert ({frozenset(p.index for p in pf.factors) for pf in eop.coefficients}
            == {frozenset([i0, i1, i2]) for i0, i1, i2 in product([0, 1, 2], [3, 4, 5], [6, 7])})


def test_expand_matrix_operator():
    # MWE from #1656
    op = expand(NumpyMatrixOperator(np.zeros((0, 0))))
    assert isinstance(op, NumpyMatrixOperator)


def test_expand_adjoint_operator_over_lincomb():
    # (c1 A + c2 B)^* == conj(c1) A^* + conj(c2) B^*
    A = NumpyMatrixOperator(np.array([[1, 2], [0, 1]], dtype=complex))
    B = NumpyMatrixOperator(np.array([[0, 1], [-1, 0]], dtype=complex))
    c1 = 2 + 3j
    c2 = -1j
    L = LincombOperator([A, B], [c1, c2])
    E = expand(AdjointOperator(L))

    assert isinstance(E, LincombOperator)
    assert all(isinstance(o, AdjointOperator) for o in E.operators)
    assert np.allclose(E.coefficients[0], np.conjugate(c1))
    assert np.allclose(E.coefficients[1], np.conjugate(c2))

    # action check on a vector
    rng = np.random.default_rng(42)
    x = A.source.from_numpy(rng.standard_normal(A.source.dim) + 1j*rng.standard_normal(A.source.dim))
    L_applied_x = AdjointOperator(L).apply(x)
    E_applied_x = E.apply(x)
    assert np.allclose(L_applied_x.to_numpy(), E_applied_x.to_numpy())


def test_adjoint_distributes_over_concatenations():
    # (A @ B @ C)^* == C^* @ B^* @ A^*
    A = NumpyMatrixOperator(np.array([[1, 2, 0], [0, 1, 0], [0, 0, 3]]))
    B = NumpyMatrixOperator(np.array([[0, 1], [-1, 0], [0, 0]]))
    C = NumpyMatrixOperator(np.array([[2, 0], [0, 4]]))

    range_product = NumpyMatrixOperator(np.eye(3))
    source_product = NumpyMatrixOperator(np.eye(2))

    concat = ConcatenationOperator([A, B, C])
    adj = AdjointOperator(concat, source_product=source_product, range_product=range_product)
    E = expand(adj)

    assert isinstance(E, ConcatenationOperator)
    ops = E.operators
    assert len(ops) == 3
    assert isinstance(ops[0], AdjointOperator)
    assert np.allclose(ops[0].operator.matrix, C.matrix)
    assert np.allclose(ops[1].matrix.T, B.matrix)
    assert isinstance(ops[2], AdjointOperator)
    assert np.allclose(ops[2].operator.matrix, A.matrix)

    # action check
    rng = np.random.default_rng(422)
    x = A.source.from_numpy(rng.standard_normal(A.source.dim) + 1j*rng.standard_normal(A.source.dim))
    adj_applied_x = adj.apply(x).to_numpy()
    E_applied_x = E.apply(x).to_numpy()
    assert np.allclose(adj_applied_x, E_applied_x)


def test_adjoints_over_concatentations_of_len_one():
    A = NumpyMatrixOperator(np.array([[1, 2, 0], [0, 1, 0], [0, 0, 3]]))
    range_product = NumpyMatrixOperator(np.eye(3))
    source_product = NumpyMatrixOperator(np.eye(3))

    adj = AdjointOperator(ConcatenationOperator([A]), source_product=source_product, range_product=range_product)
    E = expand(adj)

    assert isinstance(E, AdjointOperator)
    assert isinstance(E.operator, NumpyMatrixOperator)
    assert np.allclose(E.operator.matrix, A.matrix)

    # action check
    rng = np.random.default_rng(321)
    x = A.source.from_numpy(rng.standard_normal(A.source.dim) + 1j*rng.standard_normal(A.source.dim))
    adj_applied_x = adj.apply(x).to_numpy()
    E_applied_x = E.apply(x).to_numpy()
    assert np.allclose(adj_applied_x, E_applied_x)


def test_adjoints_over_numpymatrixoperator():
    A = NumpyMatrixOperator(np.array([[1, 2, 0], [0, 1, 0], [0, 0, 3]]))
    range_product = NumpyMatrixOperator(np.eye(3))
    source_product = NumpyMatrixOperator(np.eye(3))

    adj = AdjointOperator(A, source_product=source_product, range_product=range_product)
    E = expand(adj)

    assert isinstance(E, AdjointOperator)
    assert isinstance(E.operator, NumpyMatrixOperator)
    assert np.allclose(E.operator.matrix, A.matrix)

    # action check
    rng = np.random.default_rng(321)
    x = A.source.from_numpy(rng.standard_normal(A.source.dim) + 1j*rng.standard_normal(A.source.dim))
    adj_applied_x = adj.apply(x).to_numpy()
    E_applied_x = E.apply(x).to_numpy()
    assert np.allclose(adj_applied_x, E_applied_x)


def test_block_lincomb_expand():
    # Block [[L, D], [Z, E]], where L = a*A + pf*B
    V2 = NumpyVectorSpace(2)
    V3 = NumpyVectorSpace(3)

    A = NumpyMatrixOperator(np.array([[1, 0], [0, 2]]))
    B = NumpyMatrixOperator(np.array([[0, 1], [1, 0]]))
    D = NumpyMatrixOperator(np.array([[1, 0, 0], [0, 1, 0]]))
    Z = None
    E = NumpyMatrixOperator(np.array([[3, 1, 0], [0, 1, 0], [0, 0, 2]]))

    a = 2.0
    pf = ProjectionParameterFunctional('mu', size=1, index=0)

    L_block = LincombOperator([A, B], [a, pf])
    blocks = np.array([[L_block, D], [Z, E]], dtype=object)
    bop = BlockOperator(blocks)

    # Expand
    bop_exp = expand(bop)

    assert isinstance(bop_exp, LincombOperator)
    assert len(bop_exp.operators) == 3

    non_lincomb_blocks = bop_exp.operators[0]
    a_lincomb_block = bop_exp.operators[1]
    pf_op = bop_exp.operators[2]

    assert np.isclose(bop_exp.coefficients[0], 1.0)
    assert np.isclose(bop_exp.coefficients[1], a)
    assert bop_exp.coefficients[2] is pf

    # constant part
    assert isinstance(non_lincomb_blocks, BlockOperator)
    assert isinstance(non_lincomb_blocks.blocks[0, 0], ZeroOperator)
    assert isinstance(non_lincomb_blocks.blocks[0, 1], NumpyMatrixOperator)
    assert isinstance(non_lincomb_blocks.blocks[1, 0], ZeroOperator)
    assert isinstance(non_lincomb_blocks.blocks[1, 1], NumpyMatrixOperator)
    assert np.allclose(non_lincomb_blocks.blocks[0, 1].matrix, D.matrix)
    assert np.allclose(non_lincomb_blocks.blocks[1, 1].matrix, E.matrix)

    # "a" coefficient part
    assert isinstance(a_lincomb_block, BlockOperator)
    assert isinstance(a_lincomb_block.blocks[0, 0], NumpyMatrixOperator)
    assert isinstance(a_lincomb_block.blocks[1, 0], ZeroOperator)
    assert isinstance(a_lincomb_block.blocks[0, 1], ZeroOperator)
    assert isinstance(a_lincomb_block.blocks[1, 1], ZeroOperator)
    assert np.allclose(a_lincomb_block.blocks[0, 0].matrix, A.matrix)

    # parameter functional part
    assert isinstance(pf_op, BlockOperator)
    assert isinstance(pf_op.blocks[0, 0], NumpyMatrixOperator)
    assert isinstance(pf_op.blocks[1, 0], ZeroOperator)
    assert isinstance(pf_op.blocks[0, 1], ZeroOperator)
    assert isinstance(pf_op.blocks[1, 1], ZeroOperator)
    assert np.allclose(pf_op.blocks[0, 0].matrix, B.matrix)

    # action check
    mu = Mu({'mu': np.array([3.5])})
    block_space = BlockVectorSpace([V2, V3])
    x0 = V2.from_numpy(np.array([1, 2]))
    x1 = V3.from_numpy(np.array([3, 4, 5]))
    U = block_space.make_array([x0, x1])

    y_orig = bop.apply(U, mu=mu)
    y_exp = bop_exp.apply(U, mu=mu)
    assert np.allclose(y_orig.blocks[0].to_numpy(), y_exp.blocks[0].to_numpy())
    assert np.allclose(y_orig.blocks[1].to_numpy(), y_exp.blocks[1].to_numpy())


def test_block_expand_only_constant_blocks():
    # Block [[L, D], [Z, E]], where L = 2*A + 1*B
    V2 = NumpyVectorSpace(2)
    V3 = NumpyVectorSpace(3)

    G = NumpyMatrixOperator(np.array([[1, 0], [0, 2]]))
    F = NumpyMatrixOperator(np.array([[1, 0], [0, 2]]))
    A = LincombOperator([F, G], [3, 4])

    B = NumpyMatrixOperator(np.array([[0, 1], [1, 0]]))
    D = NumpyMatrixOperator(np.array([[1, 0, 0], [0, 1, 0]]))
    Z = None
    E = NumpyMatrixOperator(np.array([[3, 1, 0], [0, 1, 0], [0, 0, 2]]))

    L_block = LincombOperator([A, B], [2, 1])
    blocks = np.array([[L_block, D], [Z, E]], dtype=object)
    bop = BlockOperator(blocks)

    # Expand
    const_op = expand(bop)
    assert isinstance(const_op, LincombOperator)
    assert len(const_op.operators) == 4

    non_lincomb_part = const_op.operators[0]
    assert isinstance(non_lincomb_part, BlockOperator)
    assert isinstance(non_lincomb_part.blocks[0, 0], ZeroOperator)
    assert isinstance(non_lincomb_part.blocks[0, 1], NumpyMatrixOperator)
    assert isinstance(non_lincomb_part.blocks[1, 1], NumpyMatrixOperator)
    assert isinstance(non_lincomb_part.blocks[1, 0], ZeroOperator)

    first_lincomb_block = const_op.operators[1]
    assert isinstance(first_lincomb_block, BlockOperator)
    assert isinstance(first_lincomb_block.blocks[0, 0], NumpyMatrixOperator)
    assert np.allclose(first_lincomb_block.blocks[0, 0].matrix, F.matrix)
    assert np.allclose(const_op.coefficients[1], 6)
    assert isinstance(first_lincomb_block.blocks[1, 0], ZeroOperator)
    assert isinstance(first_lincomb_block.blocks[0, 1], ZeroOperator)
    assert isinstance(first_lincomb_block.blocks[1, 1], ZeroOperator)

    second_lincomb_block = const_op.operators[2]
    assert isinstance(second_lincomb_block, BlockOperator)
    assert isinstance(second_lincomb_block.blocks[0, 0], NumpyMatrixOperator)
    assert np.allclose(second_lincomb_block.blocks[0, 0].matrix, G.matrix)
    assert np.allclose(const_op.coefficients[2], 8)
    assert isinstance(second_lincomb_block.blocks[1, 0], ZeroOperator)
    assert isinstance(second_lincomb_block.blocks[0, 1], ZeroOperator)
    assert isinstance(second_lincomb_block.blocks[1, 1], ZeroOperator)

    third_lincomb_block = const_op.operators[3]
    assert isinstance(third_lincomb_block, BlockOperator)
    assert isinstance(third_lincomb_block.blocks[0, 0], NumpyMatrixOperator)
    assert np.allclose(third_lincomb_block.blocks[0, 0].matrix, B.matrix)
    assert np.allclose(const_op.coefficients[3], 1)
    assert isinstance(third_lincomb_block.blocks[1, 0], ZeroOperator)
    assert isinstance(third_lincomb_block.blocks[0, 1], ZeroOperator)
    assert isinstance(third_lincomb_block.blocks[1, 1], ZeroOperator)

    # action check
    block_space = BlockVectorSpace([V2, V3])
    x0 = V2.from_numpy(np.array([1, 2]))
    x1 = V3.from_numpy(np.array([3, 4, 5]))
    U = block_space.make_array([x0, x1])

    y_orig = bop.apply(U)
    y_exp = const_op.apply(U)
    assert np.allclose(y_orig.blocks[0].to_numpy(), y_exp.blocks[0].to_numpy())
    assert np.allclose(y_orig.blocks[1].to_numpy(), y_exp.blocks[1].to_numpy())


def test_block_expand_and_contract_only_constant_blocks():
    # Block [[L, D], [Z, E]], where L = 2*A + 1*B
    G = NumpyMatrixOperator(np.array([[1, 0], [0, 2]]))
    F = NumpyMatrixOperator(np.array([[1, 0], [0, 2]]))
    A = LincombOperator([F, G], [3, 4])

    B = NumpyMatrixOperator(np.array([[0, 1], [1, 0]]))
    D = NumpyMatrixOperator(np.array([[1, 0, 0], [0, 1, 0]]))
    Z = None
    E = NumpyMatrixOperator(np.array([[3, 1, 0], [0, 1, 0], [0, 0, 2]]))

    L_block = LincombOperator([A, B], [2, 1])
    blocks = np.array([[L_block, D], [Z, E]], dtype=object)
    bop = BlockOperator(blocks)

    # Expand and contract
    const_op = expand(bop)
    const_op_contracted = contract(const_op)
    assert isinstance(const_op_contracted, BlockOperator)
    assert isinstance(const_op_contracted.blocks[0, 0], NumpyMatrixOperator)
    assert np.allclose(const_op_contracted.blocks[0, 0].matrix, 6*F.matrix + 8*G.matrix + 1*B.matrix)


def test_contract():
    ops = [NumpyMatrixOperator(np.eye(1) * i) for i in range(1, 6)]
    pf = ProjectionParameterFunctional('p', 1, 0)

    op = (ops[0] * pf) @ (ops[1] + ops[2]) @ ops[3] @ (ops[4] * pf)

    U = op.source.ones(1)
    mu = op.parameters.parse(1)

    op_contracted = contract(op)
    assert np.all(almost_equal(op.apply(U, mu), op_contracted.apply(U, mu)))
    assert isinstance(op_contracted, ConcatenationOperator)
    assert len(op_contracted.operators) == 3
