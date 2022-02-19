# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import StationaryRBReductor
from pymortests.base import runmodule, assert_all_almost_equal


def test_ei_restricted_to_full(stationary_models):
    model = stationary_models
    op = model.operator
    cb = op.range.from_numpy(np.eye(op.range.dim))
    dofs = list(range(cb.dim))
    ei_op = EmpiricalInterpolatedOperator(op, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)
    ei_model = model.with_(operator=ei_op)

    for mu in model.parameters.space(1, 2).sample_randomly(3, seed=234):
        a = model.solve(mu)
        b = ei_model.solve(mu)
        assert_all_almost_equal(a, b, rtol=1e-4)


def test_ei_op_creation(operator):
    op = operator[0]
    cb = op.range.from_numpy(np.eye(op.range.dim))
    dofs = list(range(cb.dim))
    EmpiricalInterpolatedOperator(op, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)


def test_ei_rom(stationary_models):
    fom = stationary_models
    op = fom.operator
    cb = op.range.from_numpy(np.eye(op.range.dim))
    dofs = list(range(cb.dim))
    ei_op = EmpiricalInterpolatedOperator(op, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)
    ei_fom = fom.with_(operator=ei_op)

    U = fom.solution_space.empty()
    base_mus = []
    for mu in fom.parameters.space(1, 2).sample_randomly(3, seed=234):
        a = fom.solve(mu)
        U.append(a)
        base_mus.append(mu)

    rb, svals = pod(U, rtol=1e-7)
    reductor = StationaryRBReductor(ei_fom, rb)
    rom = reductor.reduce()
    for mu, u in zip(base_mus, U):
        ru = rom.solve(mu)
        ru_rec = reductor.reconstruct(ru)
        assert_all_almost_equal(u, ru_rec, rtol=1e-10)


if __name__ == "__main__":
    runmodule(filename=__file__)
