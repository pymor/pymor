# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.system import project_system
from pymor.operators.constructions import ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.block import BlockOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.gram_schmidt import gram_schmidt


def test_project_system():
    A, B, C, D = (
        NumpyMatrixOperator(np.eye(5)), NumpyMatrixOperator(np.eye(5) * 2), None, NumpyMatrixOperator(np.eye(5) * 3)
    )
    op = BlockOperator([[A, B], [C, D]])
    U, V = gram_schmidt(NumpyVectorSpace(5).random(2, seed=123)), gram_schmidt(NumpyVectorSpace(5).random(3, seed=456))
    pop = project_system(op, [U, V], [U, V])
    a, b, c, d = pop.blocks.ravel()
    assert np.max(np.abs(a.matrix - np.eye(2))) < 1e-15
    assert np.max(np.abs(b.matrix - U.inner(V) * 2)) < 1e-15
    assert isinstance(c, ZeroOperator)
    assert np.max(np.abs(d.matrix - np.eye(3)*3)) < 1e-15
