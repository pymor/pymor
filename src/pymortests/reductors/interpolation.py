# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.interpolation import TFBHIReductor

pytestmark = pytest.mark.builtin


def test_tfbhi_value_and_derivative_interpolation():
    fom = TransferFunction(
        1,
        1,
        lambda s: np.array([[sum((i + 1) / (s + i + 1) for i in range(5))]]),
        lambda s: np.array([[-sum((i + 1) / (s + i + 1)**2 for i in range(5))]]),
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
