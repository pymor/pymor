# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel, PHLTIModel
from pymor.operators.constructions import IdentityOperator
from pymor.reductors.ph.ph_irka import PHIRKAReductor

pytestmark = pytest.mark.builtin


def test_ph_irka():
    J = np.array([[0, 1], [-1, 0]])
    R = np.array([[1, 0], [0, 1]])
    G = np.array([[1], [0]])
    fom = PHLTIModel.from_matrices(J, R, G)
    phirka = PHIRKAReductor(fom)

    rom = phirka.reduce(1)
    assert isinstance(rom, PHLTIModel) and rom.order == 1

    rom = phirka.reduce(np.array([1]))
    assert isinstance(rom, PHLTIModel) and rom.order == 1

    rom = phirka.reduce({'sigma': np.array([1]),
                         'b': np.array([[1]]),
                         'c': np.array([[1]])})
    assert isinstance(rom, PHLTIModel) and rom.order == 1

    Ar = np.array([-1])
    Br = np.array([1])
    Cr = np.array([1])
    initial_rom = LTIModel.from_matrices(Ar, Br, Cr)
    rom = phirka.reduce(initial_rom)
    assert isinstance(rom, PHLTIModel) and rom.order == 1

    Jr = np.array([0])
    Rr = np.array([1])
    Gr = np.array([1])
    initial_rom = PHLTIModel.from_matrices(Jr, Rr, Gr)
    rom = phirka.reduce(initial_rom)
    assert isinstance(rom, PHLTIModel) and rom.order == 1

def test_ph_irka_E_and_Q():
    J = np.array([[0, 1], [-1, 0]])
    R = np.array([[1, 0], [0, 1]])
    G = np.array([[1], [0]])
    E = np.array([[2, 0], [0, 1]])
    Q = np.array([[5, 0], [0, 3]])
    fom = PHLTIModel.from_matrices(J, R, G, E=E, Q=Q)
    phirka = PHIRKAReductor(fom)

    rom1 = phirka.reduce(1)
    assert isinstance(rom1, PHLTIModel) and rom1.order == 1
    Er = to_matrix(rom1.E, format='dense')
    assert not np.isclose(Er, np.array([[1]]))
    assert np.isclose(Er, fom.E.apply2(phirka.W, phirka.V))

    rom2 = phirka.reduce(1, projection='QTEorth')
    assert isinstance(rom2, PHLTIModel) and rom2.order == 1
    assert isinstance(rom2.E, IdentityOperator)
    assert np.isclose(np.array([[1]]), fom.E.apply2(phirka.W, phirka.V))

    err1 = (rom1 - fom).h2_norm()
    err2 = (rom2 - fom).h2_norm()
    assert abs(err1 - err2) < 1e-12
