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


@pytest.mark.parametrize(('dim_input', 'dim_output'), [(1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize('complex_data', [False, True])
@pytest.mark.parametrize('varying_factor', ['K', 'B', 'C', 'D', 'all'])
def test_factorized_transfer_function_derivative(dim_input, dim_output, complex_data, varying_factor, rng):
    shapes = {'K': (3, 3), 'B': (3, dim_input), 'C': (dim_output, 3), 'D': (dim_output, dim_input)}
    constants, derivatives = {}, {}
    for name, shape in shapes.items():
        constant = rng.standard_normal(shape)
        derivative = rng.standard_normal(shape)
        if complex_data:
            constant = constant + 1j * rng.standard_normal(shape)
            derivative = derivative + 1j * rng.standard_normal(shape)
        constants[name] = constant
        derivatives[name] = derivative if varying_factor in (name, 'all') else np.zeros(shape)
    constants['K'] += 4 * np.eye(3)

    s = ProjectionParameterFunctional('s')
    operators = {name: (NumpyMatrixOperator(constants[name]) + s * NumpyMatrixOperator(derivatives[name])
                        if varying_factor in (name, 'all') else NumpyMatrixOperator(constants[name]))
                 for name in shapes}
    derivative_operators = {f'd{name}': NumpyMatrixOperator(derivatives[name]) for name in shapes}
    tf = FactorizedTransferFunction(dim_input, dim_output, **operators, **derivative_operators)

    for frequency in (0, 1j, 0.7 + 0.2j):
        if dim_input > dim_output and varying_factor in ('C', 'all') and np.imag(frequency):
            with pytest.raises(NotImplementedError, match='non-real coefficients'):
                tf.eval_tf(frequency)
            with pytest.raises(NotImplementedError, match='non-real coefficients'):
                tf.eval_dtf(frequency)
            continue
        K, B, C, D = (constants[name] + frequency * derivatives[name] for name in shapes)
        dK, dB, dC, dD = (derivatives[name] for name in shapes)
        X = np.linalg.solve(K, B)
        expected_derivative = dC @ X + C @ np.linalg.solve(K, dB - dK @ X) + dD
        np.testing.assert_allclose(tf.eval_tf(frequency), C @ X + D)
        np.testing.assert_allclose(tf.eval_dtf(frequency), expected_derivative)


@pytest.mark.parametrize(('dim_input', 'dim_output'), [(1, 2), (2, 1)])
@pytest.mark.parametrize('degree', [1, 2])
@pytest.mark.parametrize('combined', [False, True])
def test_factorized_transfer_function_complex_coefficient_guard(dim_input, dim_output, degree, combined):
    s = ProjectionParameterFunctional('s')
    K = NumpyMatrixOperator(np.eye(2))
    B = NumpyMatrixOperator(np.ones((2, dim_input)))
    C = NumpyMatrixOperator((1. + 2j) * np.ones((dim_output, 2)))
    D = NumpyMatrixOperator(np.zeros((dim_output, dim_input)))
    tf = FactorizedTransferFunction(
        dim_input, dim_output, K, B, (s if degree == 1 else s * s) * C, D,
        dK=0 * K, dB=0 * B, dC=C if degree == 1 else 2 * s * C, dD=0 * D,
    )
    if combined:
        tf = tf + tf
    H = (2 if combined else 1) * C.matrix @ B.matrix

    if dim_input > dim_output and degree == 1:
        with pytest.raises(NotImplementedError, match='non-real coefficients'):
            tf.eval_tf(1j)
    else:
        np.testing.assert_allclose(tf.eval_tf(1j), 1j**degree * H)
    if dim_input > dim_output:
        with pytest.raises(NotImplementedError, match='non-real coefficients'):
            tf.eval_dtf(1j)
    else:
        np.testing.assert_allclose(tf.eval_dtf(1j), degree * 1j**(degree - 1) * H)
