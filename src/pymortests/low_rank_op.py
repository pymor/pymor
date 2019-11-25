# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.lincomb import assemble_lincomb
from pymor.operators.constructions import LowRankOperator, LowRankUpdatedOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test_low_rank_assemble():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)

    r1 = 2
    L1 = space.random(r1, distribution='normal', random_state=rng)
    C1 = rng.randn(r1, r1)
    R1 = space.random(r1, distribution='normal', random_state=rng)

    r2 = 3
    L2 = space.random(r2, distribution='normal', random_state=rng)
    C2 = rng.randn(r2, r2)
    R2 = space.random(r2, distribution='normal', random_state=rng)

    LR1 = LowRankOperator(L1, C1, R1)
    LR2 = LowRankOperator(L2, C2, R2)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert isinstance(op, LowRankOperator)
    assert len(op.left) == r1 + r2
    assert not op.inverted

    op = (LR1 + (LR1 + LR2) + LR2).assemble()
    assert isinstance(op, LowRankOperator)

    LR1 = LowRankOperator(L1, C1, R1, inverted=True)
    LR2 = LowRankOperator(L2, C2, R2, inverted=True)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert isinstance(op, LowRankOperator)
    assert len(op.left) == r1 + r2
    assert op.inverted

    LR1 = LowRankOperator(L1, C1, R1, inverted=True)
    LR2 = LowRankOperator(L2, C2, R2)
    op = assemble_lincomb([LR1, LR2], [1, 1])
    assert op is None


def test_low_rank_updated_assemble():
    n = 10
    space = NumpyVectorSpace(n)
    rng = np.random.RandomState(0)
    A = NumpyMatrixOperator(rng.randn(n, n))

    r = 2
    L = space.random(r, distribution='normal', random_state=rng)
    C = rng.randn(r, r)
    R = space.random(r, distribution='normal', random_state=rng)
    LR = LowRankOperator(L, C, R)

    op = (A + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)

    op = (A + LR + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)

    op = (A + (A + LR).assemble() + LR).assemble()
    assert isinstance(op, LowRankUpdatedOperator)
