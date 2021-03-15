# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.basic import StationaryModel
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymortests.base import runmodule, assert_all_almost_equal
from pymortests.fixtures.operator import operator
from pymortests.fixtures.model import model, stationary_models


def test_ei_restricted_to_full(stationary_models):
    model = stationary_models
    op = model.operator
    cb = op.range.from_numpy(np.eye(op.range.dim))
    dofs = list(range(cb.dim))
    ei_op = EmpiricalInterpolatedOperator(op, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)
    ei_model = StationaryModel(ei_op, model.rhs)

    for mu in model.parameters.space(1, 2).sample_randomly(3, seed=234):
        a = model.solve(mu)
        b = ei_model.solve(mu)
        assert_all_almost_equal(a, b, rtol=1e-4)


def test_ei_op_creation(operator):
    op = operator[0]
    cb = op.range.from_numpy(np.eye(op.range.dim))
    dofs = list(range(cb.dim))
    EmpiricalInterpolatedOperator(op, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)


if __name__ == "__main__":
    runmodule(filename=__file__)
