# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import product

import numpy as np

from pymor.algorithms.to_matrix import to_matrix
from pymor.algorithms.simplify import expand
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
