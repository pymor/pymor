# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.phdmd import phdmd
from pymor.core.exceptions import PHDMDError
from pymor.models.examples import msd_example
from pymor.models.iosys import PHLTIModel
from pymordemos.phdmd import _implicit_midpoint, excitation_control


def _get_fitting_data(order, num_outputs=1, initial_J=False, initial_R=False):
    assert not (initial_J and initial_R)

    J, R, G, P, S, N, E, Q = msd_example(order, m=num_outputs)
    fom = PHLTIModel.from_matrices(J, R, G, P, S, N, E, Q).to_berlin_form()

    time_stamps = np.linspace(0, 4, 100)
    dt = time_stamps[1]
    fom_state_space = fom.solution_space
    fom_initial = np.zeros(fom_state_space.dim)

    fom_X, U, fom_Y, _ = _implicit_midpoint(fom, excitation_control, fom_initial, time_stamps)

    op = None
    if initial_J:
        op = np.block([
            [Q.T @ J @ Q, Q.T @ G],
            [-G.T @ Q, N]
        ])
    if initial_R:
        op = np.block([
            [Q.T @ R @ Q, Q.T @ P],
            [P.T @ Q, S]
        ])
    return fom_X, U, fom_Y, E.T @ Q, dt, op


@pytest.mark.parametrize('order', list(range(6, 21, 4)))
@pytest.mark.parametrize('rtol', [1e-7, 1e-10])
def test_phdmd_no_init(order, rtol):
    X, U, Y, H, dt, _ = _get_fitting_data(order)
    try:
        _, data = phdmd(X, U, Y, dt=dt, H=H, rtol=rtol)
        assert data['rel_errs'][-1] < rtol or data['update_norms'][-1] < rtol
    except PHDMDError:
        # increasing model order reduces approximation for the same time stamps
        assert rtol == 1e-10 and order > 12

@pytest.mark.parametrize('order', list(range(6, 21, 4)))
@pytest.mark.parametrize('rtol', [1e-6, 1e-8])
def test_phdmd_J_init(order, rtol):
    X, U, Y, H, dt, initial_J = _get_fitting_data(order, initial_J=True)
    try:
        _, data = phdmd(X, U, Y, dt=dt, H=H, rtol=rtol, initial_J=initial_J)
        assert data['rel_errs'][-1] < rtol or data['update_norms'][-1] < rtol
    except PHDMDError:
        # increasing model order reduces approximation for the same time stamps
        assert order > 16

@pytest.mark.parametrize('order', list(range(6, 21, 4)))
@pytest.mark.parametrize('rtol', [1e-8, 1e-10])
def test_phdmd_R_init(order, rtol):
    X, U, Y, H, dt, initial_R = _get_fitting_data(order, initial_R=True)
    _, data = phdmd(X, U, Y, dt=dt, H=H, rtol=rtol, initial_R=initial_R)
    assert data['rel_errs'][-1] < rtol or data['update_norms'][-1] < rtol
