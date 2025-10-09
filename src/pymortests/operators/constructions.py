# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.operators.constructions import AdjointOperator, InverseOperator
from pymor.operators.numpy import NumpyMatrixOperator


def test_adjoint_operator():
    op = NumpyMatrixOperator(np.array([[1., 1.], [1., 0]]))
    sp = NumpyMatrixOperator(np.diag([1., 2.]))
    rp = NumpyMatrixOperator(np.diag([4., 5.]))

    adj_op = AdjointOperator(op, range_product=rp, source_product=sp)
    U = adj_op.source.from_numpy(np.array([1., 0]))

    assert np.all(almost_equal((InverseOperator(sp) @ op.H @ rp).apply(U),
                               adj_op.apply(U)))
    assert np.all(almost_equal((rp @ op @ InverseOperator(sp)).apply(U),
                               adj_op.apply_adjoint(U)))
    assert np.all(almost_equal(adj_op.H.apply(U), adj_op.apply_adjoint(U)))
