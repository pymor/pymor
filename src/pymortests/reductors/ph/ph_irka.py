# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.iosys import LTIModel, PHLTIModel
from pymor.reductors.ph.ph_irka import PHIRKAReductor


def test_irka():
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
