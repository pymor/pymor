# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import pytest
import numpy as np
from hypothesis import given

import pymortests.strategies as pyst
from pymor.algorithms.bfgs import bfgs
from pymor.core.exceptions import BFGSError
from pymordemos.linear_optimization import create_fom
from pymor.algorithms.bfgs import get_active_and_inactive_sets


@given(pyst.active_mu_data())
def test_active_inactive_sets(data):
    space, mus, active_indices = data
    dim = space.parameters.dim
    for it in range(len(mus)):
        mu = mus[it]
        active_inds = active_indices[it]

        active, inactive = get_active_and_inactive_sets(space, mu)
        compare = np.zeros(dim)
        compare[active_inds] = 1

        assert all(compare == active)
        assert all(np.ones(dim) - compare == inactive)


def test_bfgs():
    fom, _ = create_fom(10)
    parameter_space = fom.parameters.space(0, np.pi)
    initial_guess = fom.parameters.parse([0.25, 0.5])

    # successful run
    _, data = bfgs(fom, parameter_space, initial_guess=initial_guess)
    assert len(data['mus']) == 17

    # failing run
    with pytest.raises(BFGSError):
        mu, _ = bfgs(fom, parameter_space, initial_guess=initial_guess, maxiter=10)
