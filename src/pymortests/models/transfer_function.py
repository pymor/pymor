# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.models.transfer_function import FactorizedTransferFunction
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize(('dim_input', 'dim_output'), [(1, 2), (2, 1)])
def test_factorized_transfer_function_complex_feedthrough(dim_input, dim_output):
    s = ProjectionParameterFunctional('s')
    K = NumpyMatrixOperator(np.eye(2))
    B = NumpyMatrixOperator(np.ones((2, dim_input)))
    C = NumpyMatrixOperator(np.ones((dim_output, 2)))
    D = NumpyMatrixOperator(np.ones((dim_output, dim_input)))
    tf = FactorizedTransferFunction(
        dim_input, dim_output, K, B, C, s * s * D,
        dK=0 * K, dB=0 * B, dC=0 * C, dD=2 * s * D,
    )

    np.testing.assert_allclose(tf.eval_tf(1j), C.matrix @ B.matrix - D.matrix)
    np.testing.assert_allclose(tf.eval_dtf(1j), 2j * D.matrix)
