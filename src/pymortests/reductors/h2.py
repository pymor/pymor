# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel
from pymor.reductors.h2 import IRKAReductor, TFIRKAReductor

pytestmark = pytest.mark.builtin


def test_irka():
    A = np.array([[-1, 0], [0, -2]])
    B = np.array([[1], [2]])
    C = np.array([[2, 1]])
    fom = LTIModel.from_matrices(A, B, C)
    irka = IRKAReductor(fom)

    rom = irka.reduce(1)
    assert isinstance(rom, LTIModel)
    assert rom.order == 1

    rom = irka.reduce(np.array([1]))
    assert isinstance(rom, LTIModel)
    assert rom.order == 1

    rom = irka.reduce({'sigma': np.array([1]),
                       'b': np.array([[1]]),
                       'c': np.array([[1]])})
    assert isinstance(rom, LTIModel)
    assert rom.order == 1

    rom = irka.reduce(LTIModel.from_matrices(np.array([[-1]]),
                                             np.array([[1]]),
                                             np.array([[1]])))
    assert isinstance(rom, LTIModel)
    assert rom.order == 1


def test_tfirka_numerical_conjugate_pairs():
    blocks = [np.array([[-damping, frequency], [-frequency, -damping]])
              for damping, frequency in [(0.05, 2), (0.1, 5), (0.2, 9)]]
    A = np.block([[blocks[i] if i == j else np.zeros((2, 2)) for j in range(3)] for i in range(3)])
    B = np.array([[1], [0.3], [0.5], [1.2], [-0.4], [0.8]])
    C = np.array([[0.7, -0.2, 1, 0.4, 0.2, -0.8]])
    fom = LTIModel.from_matrices(A, B, C)
    sigma = np.array([0.1 + 2j, 0.1 - 2j, 0.2 + 5j, 0.2 - 5j])
    initial_data = {'sigma': sigma, 'b': np.ones((4, 1)), 'c': np.ones((4, 1))}

    rom = TFIRKAReductor(fom).reduce(initial_data, maxit=2, tol=1e-12)

    assert all(np.isrealobj(to_matrix(getattr(rom, operator), format='dense')) for operator in 'ABCE')
