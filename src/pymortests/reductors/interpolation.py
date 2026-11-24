# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.interpolation import LTIBHIReductor, TFBHIReductor

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize('projection', ['orth', 'biorth'])
def test_ltibhi_mimo_tangential_interpolation(projection, rng):
    n, m, p = 10, 3, 2
    A = -np.diag(np.arange(1, n + 1)) + 0.1 * rng.standard_normal((n, n))
    fom = LTIModel.from_matrices(A, rng.standard_normal((n, m)), rng.standard_normal((p, n)))
    sigma = np.array([1, 2 + 3j, 2 - 3j])
    b1 = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    c1 = rng.standard_normal(p) + 1j * rng.standard_normal(p)
    b = np.array([rng.standard_normal(m), b1, b1.conj()])
    c = np.array([rng.standard_normal(p), c1, c1.conj()])

    rom = LTIBHIReductor(fom).reduce(sigma, b, c, projection=projection)

    for s, bi, ci in zip(sigma, b, c, strict=True):
        H, Hr = fom.transfer_function.eval_tf(s), rom.transfer_function.eval_tf(s)
        dH, dHr = fom.transfer_function.eval_dtf(s), rom.transfer_function.eval_dtf(s)
        assert np.allclose(H @ bi, Hr @ bi)
        assert np.allclose(ci @ H, ci @ Hr)
        assert np.allclose(ci @ dH @ bi, ci @ dHr @ bi)


@pytest.mark.parametrize('projection', ['orth', 'biorth', 'arnoldi'])
def test_ltibhi_parametric(projection):
    n = 10
    p = ProjectionParameterFunctional('p')
    A = NumpyMatrixOperator(-np.diag(np.arange(1, n + 1))) + p * NumpyMatrixOperator(-np.eye(n))
    B = NumpyMatrixOperator(np.ones((n, 1)))
    C = NumpyMatrixOperator(np.arange(1, n + 1)[np.newaxis])
    E = NumpyMatrixOperator(np.eye(n)) + p * NumpyMatrixOperator(0.1 * np.eye(n))
    fom = LTIModel(A, B, C, E=E)
    mu = fom.parameters.parse(5)
    sigma = np.array([1, 2 + 3j, 2 - 3j])
    ones = np.ones((3, 1))

    rom = LTIBHIReductor(fom, mu=mu).reduce(sigma, ones, ones, projection=projection)

    for s in sigma:
        assert np.allclose(rom.transfer_function.eval_tf(s), fom.transfer_function.eval_tf(s, mu=mu))
        assert np.allclose(rom.transfer_function.eval_dtf(s), fom.transfer_function.eval_dtf(s, mu=mu))


def test_tfbhi_value_and_derivative_interpolation():
    fom = TransferFunction(
        1,
        1,
        lambda s: np.array([[sum((i + 1) / (s + i + 1) for i in range(5))]]),
        dtf=lambda s: np.array([[-sum((i + 1) / (s + i + 1)**2 for i in range(5))]]),
    )
    sigma = np.array([0, 1j, -1j])
    directions = np.ones((3, 1))
    values = np.array([fom.eval_tf(node).item() for node in sigma])
    scale = 1 / np.sqrt(2)
    transformation = np.array([
        [1, 0, 0],
        [0, scale, scale],
        [0, -1j * scale, 1j * scale],
    ])

    rom = TFBHIReductor(fom).reduce(sigma, directions, directions)

    assert np.allclose(to_matrix(rom.B), (transformation @ values[:, np.newaxis]).real)
    assert np.allclose(to_matrix(rom.C), (values[np.newaxis] @ transformation.conj().T).real)
    for node in sigma:
        assert np.allclose(rom.transfer_function.eval_tf(node), fom.eval_tf(node))
        assert np.allclose(rom.transfer_function.eval_dtf(node), fom.eval_dtf(node))
