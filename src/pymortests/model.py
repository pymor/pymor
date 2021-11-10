# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.basic import almost_equal
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.core.pickle import dumps, loads
from pymortests.base import runmodule
from pymortests.pickling import assert_picklable, assert_picklable_without_dumps_function


def test_pickle(model):
    assert_picklable(model)


def test_pickle_without_dumps_function(picklable_model):
    assert_picklable_without_dumps_function(picklable_model)


def test_pickle_by_solving(model):
    m = model
    m2 = loads(dumps(m))
    m.disable_caching()
    m2.disable_caching()
    for mu in m.parameters.space(1, 2).sample_randomly(3, seed=234):
        assert np.all(almost_equal(m.solve(mu), m2.solve(mu)))


def test_StationaryModel_deaffinize():

    p = thermal_block_problem((2, 2)).with_(
        dirichlet_data=ExpressionFunction('x[0]', 2),
        outputs=[('l2', ConstantFunction(1., 2))]
    )
    m, _ = discretize_stationary_cg(p, diameter=1/10)

    U_aff = m.solve([1, 1, 1, 1])
    m_deaff = m.deaffinize(U_aff)

    mu = m.parameters.parse([0.1, 10, 7, 1])

    U = m.solve(mu)
    U_deaff = m_deaff.solve(mu)
    assert np.all(almost_equal(U, U_deaff + U_aff))
    assert np.allclose(m.output(mu), m_deaff.output(mu))


if __name__ == "__main__":
    runmodule(filename=__file__)
