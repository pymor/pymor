# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from hypothesis import given

from pymor.algorithms.basic import almost_equal
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper, ExplicitEulerTimeStepper, ExplicitRungeKuttaTimeStepper
from pymor.discretizers.builtin.fv import discretize_instationary_fv

from pymortests.base import runmodule
from pymortests.fixtures.analyticalproblem import linear_transport_problems


@pytest.mark.parametrize('num_values_factor', [None, 1, 3, 10])
def test_ExplicitEuler_linear_transport_exact(num_values_factor):
    nt = 10
    fom, _ = discretize_instationary_fv(linear_transport_problems[0], diameter=1/nt, nt=nt)
    if num_values_factor:
        num_values = num_values_factor*(nt + 1)
        fom = fom.with_time_stepper(num_values=num_values)
    else:
        num_values = nt + 1
    U = fom.solve()
    assert len(U) == num_values
    assert almost_equal(U[0], U[-1])


if __name__ == "__main__":
    runmodule(filename=__file__)
