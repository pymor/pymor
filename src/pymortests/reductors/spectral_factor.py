# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest
import scipy.linalg as spla

from pymor.models.iosys import LTIModel
from pymor.reductors.h2 import IRKAReductor
from pymor.reductors.spectral_factor import SpectralFactorReductor

pytestmark = pytest.mark.builtin


def test_spectral_factor():
    R = np.array([[1, 0], [0, 2]])
    G = np.array([[1], [2]])
    S = np.array([[1e-12]])
    fom = LTIModel.from_matrices(-R, G, G.T, S, solver_options={'ricc_pos_lrcf': 'slycot'})

    Z = fom.gramian('pr_o_lrcf').to_numpy()
    X = Z.T@Z
    assert np.all(spla.eigvals(X) > 0), 'Passive FOM expected.'

    spectralFactor = SpectralFactorReductor(fom)

    rom = spectralFactor.reduce(
        lambda spectral_factor, mu : IRKAReductor(spectral_factor,mu).reduce(1))
    assert isinstance(rom, LTIModel) and rom.order == 1

    assert np.all(np.real(rom.poles()) < 0), 'Asymptotically stable ROM expected.'

    rom = rom.with_(solver_options={'ricc_pos_lrcf': 'slycot'})
    Z2 = rom.gramian('pr_o_lrcf').to_numpy()
    X2 = Z2.T@Z2
    assert np.all(spla.eigvals(X2) > 0), 'Passive ROM expected.'

def test_spectral_factor_invertible_D_assertion():
    R = np.array([[1, 0], [0, 2]])
    G = np.array([[1], [2]])
    fom = LTIModel.from_matrices(-R, G, G.T, solver_options={'ricc_pos_lrcf': 'slycot'})

    spectralFactor = SpectralFactorReductor(fom)

    with pytest.raises(AssertionError) as e:
        spectralFactor.reduce(
            lambda spectral_factor, mu : IRKAReductor(spectral_factor,mu).reduce(1))

    assert 'D+D^T must be invertible.' in str(e.value)
