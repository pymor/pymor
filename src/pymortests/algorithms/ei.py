# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.ei import ei_greedy
from pymor.algorithms.pod import pod
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.reductors.basic import StationaryRBReductor
from pymor.solvers.generic import LGMRESSolver
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymortests.base import assert_all_almost_equal, runmodule


def test_ei_restricted_to_full(stationary_models):
    model = stationary_models
    op = model.operator
    cb = op.range.from_numpy(np.eye(op.range.dim))
    dofs = list(range(cb.dim))
    ei_op = EmpiricalInterpolatedOperator(op, collateral_basis=cb, interpolation_dofs=dofs, triangular=True,
                                          solver=LGMRESSolver())
    ei_model = model.with_(operator=ei_op)

    mus = model.parameters.space(1, 2).sample_randomly(3)
    for mu in mus:
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
    mus = fom.parameters.space(1, 2).sample_randomly(3)
    for mu in mus:
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


def test_ei_greedy_complex_data():
    space = NumpyVectorSpace(10)
    U = space.random(3) * 1.j + space.random(3)
    ei_greedy(U)


if __name__ == '__main__':
    runmodule(filename=__file__)
