# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.phdmd import phdmd
from pymor.core.exceptions import PHDMDError
from pymor.models.iosys import PHLTIModel
from pymordemos.phdmd import _implicit_midpoint, excitation_control
from pymordemos.phlti import msd


def _get_fitting_data(order):
    J, R, G, P, S, N, E = msd(order)
    fom = PHLTIModel.from_matrices(J, R, G, P, S, N, E)

    dt = 4e-2
    time_stamps = np.arange(0., 4. + dt, dt)
    fom_state_space = fom.solution_space
    io_space = fom.G.source
    fom_initial = np.zeros(fom_state_space.dim)

    fom_X, U, fom_Y, _ = _implicit_midpoint(fom, excitation_control, fom_initial, time_stamps)
    fom_X = fom_state_space.from_numpy(fom_X.T)
    fom_Y = io_space.from_numpy(fom_Y.T)
    U = io_space.from_numpy(U.T)

    return fom_X, fom_Y, U, E, dt


@pytest.mark.parametrize('order', list(range(6, 21, 2)))
@pytest.mark.parametrize('rtol', [1e-8, 1e-10])
def test_phdmd(order, rtol):
    X, Y, U, H, dt = _get_fitting_data(order)
    try:
        inf_fom, _ = phdmd(X, Y, U, dt=dt, H=H, rtol=rtol)
    except PHDMDError:
        # increasing model order reduces approximation for the same time stamps
        expected = False
        if ((rtol == 1e-10 and order > 12) or
            (rtol == 1e-08 and order == 14)):
            expected = True
        assert expected
