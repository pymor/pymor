# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from unittest import mock

import numpy as np
import pytest

from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymortests.models.iosys_add_sub_mul import get_model

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('parametric', [False, True])
def test_hinf_norm(parametric):
    m = get_model('LTIModel', 0, parametric)

    if parametric:
        mu = m.parameters.parse([-1.])
    else:
        mu = None

    hinf_norm = m.hinf_norm(mu=mu)
    hinf_norm_f, fpeak = m.hinf_norm(mu=mu, return_fpeak=True)

    assert hinf_norm == hinf_norm_f


@pytest.mark.parametrize('alpha', [0., 1.])
def test_h2_inner_warns_on_nonzero_feedthrough(alpha):
    A = NumpyMatrixOperator(np.array([[-1.]]))
    B = NumpyMatrixOperator(np.ones((1, 1)))
    C = NumpyMatrixOperator(np.ones((1, 1)))
    D = ProjectionParameterFunctional('alpha') * NumpyMatrixOperator(np.ones((1, 1)))
    lti = LTIModel(A, B, C, D)
    tf = TransferFunction(1, 1, lambda s: np.array([[1 / (s + 1)]]))
    mu = lti.parameters.parse(alpha)
    with mock.patch.object(tf.logger, 'warning') as warning:
        assert np.allclose(tf.h2_inner(lti, mu=mu), 0.5)
    assert warning.called == (alpha != 0)
