# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.to_matrix import to_matrix
from pymor.algorithms.simplify import expand, contract
from pymor.operators.constructions import LincombOperator, ConcatenationOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional


def test_expand():
    ops = [NumpyMatrixOperator(np.eye(1) * i) for i in range(8)]
    pfs = [ProjectionParameterFunctional('p', 9, i) for i in range(8)]
    prods = [o * p for o, p in zip(ops, pfs)]

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
