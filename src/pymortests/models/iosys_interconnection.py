# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.examples import msd_example
from pymor.models.iosys import PHLTIModel

pytestmark = pytest.mark.builtin


def msd_via_interconnection(n=6, m=2, m_i=4, k_i=4, c_i=1):
    """Mass-spring-damper model of :func:`~pymor.models.examples.msd_example` as interconnection."""
    assert n % 2 == 0
    assert n >= 6
    assert m in (1, 2)
    num_masses = n // 2

    # ports: force from left spring, force from right spring, external force
    mass = PHLTIModel.from_matrices(np.zeros((1, 1)), np.array([[c_i]]), np.ones((1, 3)), Q=np.array([[1 / m_i]]))
    LEFT, RIGHT, FORCE = 0, 1, 2

    # spring between masses i and i+1: elongation e = q_{i+1} - q_i
    coupling_spring = PHLTIModel.from_matrices(np.zeros((1, 1)), np.zeros((1, 1)), np.array([[-1., 1.]]),
                                               Q=np.array([[k_i]]))

    wall_spring = PHLTIModel.from_matrices(np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), Q=np.array([[k_i]]))
    models = [mass] * num_masses + [coupling_spring] * (num_masses - 1) + [wall_spring]

    K = np.eye(1)
    connections = []
    for i in range(num_masses - 1):
        spring_idx = num_masses + i
        connections.append(((spring_idx, ('idx', [LEFT])), (i, ('idx', [RIGHT])), K))
        connections.append(((spring_idx, ('idx', [RIGHT])), (i + 1, ('idx', [LEFT])), K))

    connections.append(((len(models) - 1, ('idx', [LEFT])), (num_masses - 1, ('idx', [RIGHT])), K))
    external = [(i, ('idx', [FORCE])) for i in range(m)]

    return PHLTIModel.from_interconnection(models, connections, external=external)


@pytest.mark.parametrize('n', [6, 10])
@pytest.mark.parametrize('m', [1, 2])
def test_msd_via_interconnection(n, m):
    fom = msd_example(n, m)
    fom_ic = msd_via_interconnection(n, m)

    assert isinstance(fom_ic, PHLTIModel)
    assert fom_ic.order == fom.order
    assert fom_ic.dim_input == fom.dim_input
    assert fom_ic.dim_output == fom.dim_output

    J = to_matrix(fom_ic.J, format='dense')
    assert np.all(J == -J.T)

    w = np.logspace(-3, 3, 50)
    H = fom.transfer_function.freq_resp(w)
    H_ic = fom_ic.transfer_function.freq_resp(w)
    rel_err = np.linalg.norm(H - H_ic, ord=2, axis=(1, 2)) / np.linalg.norm(H, ord=2, axis=(1, 2))
    assert np.all(rel_err < 1e-10)
