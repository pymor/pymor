#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
from typer import Argument, run

from pymor.basic import *


def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the reduced basis.'),
    verification_samples: int = Argument(..., help='Number of samples used for verification of the output error.')
):
    """Example script for using the DWR output error estimation"""
    # real valued output
    fom_1 = create_fom(grid_intervals, vector_valued_output=False)

    # vector valued output (with BlockColumnOperator)
    fom_2 = create_fom(grid_intervals, vector_valued_output=True)

    # an output which is actually a lincomb operator
    dim_source = fom_1.output_functional.source.dim
    random_matrix_1 = np.random.rand(2, dim_source)
    random_matrix_2 = np.random.rand(2, dim_source)
    op1 = NumpyMatrixOperator(random_matrix_1, source_id='STATE')
    op2 = NumpyMatrixOperator(random_matrix_2, source_id='STATE')
    ops = [op1, op2]
    lincomb_op = LincombOperator(ops, [1., 0.5])
    fom_3 = fom_2.with_(output_functional=lincomb_op)

    # all foms with different output_functionals
    foms = [fom_1, fom_2, fom_3]

    # only use CoerciveRBReductor for now
    reductor = CoerciveRBReductor

    # Parameter space and operator are equal for all foms
    parameter_space = fom_1.parameters.space(0.1, 1)
    training_set = parameter_space.sample_randomly(training_samples)
    random_set = parameter_space.sample_randomly(verification_samples, seed=11)
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom_1.parameters)

    estimator_values = []
    for fom in foms:
        for operator_symmetric in [True, False]:
            for construct_dual_bases in [False, True]:
                # generate solution snapshots
                primal_snapshots = fom.solution_space.empty()
                dual_snapshotss = [fom.solution_space.empty() for d in range(fom.dim_output)]

                # construct training data
                for mu in training_set:
                    primal_snapshots.append(fom.solve(mu))

                for d in range(fom.dim_output):
                    # initialize dual model from reductor
                    dual_fom = reductor.dual_model(fom, d, operator_symmetric)
                    for mu in training_set:
                        dual_snapshotss[d].append(dual_fom.solve(mu))

                # apply POD on bases
                primal_reduced_basis = pod(primal_snapshots, modes=4)[0]
                if construct_dual_bases:
                    dual_reduced_bases = []
                    for d in range(fom.dim_output):
                        dual_reduced_bases.append(pod(dual_snapshotss[d], modes=4)[0])
                else:
                    dual_reduced_bases = None

                RB_reductor = reductor(fom,
                                       RB=primal_reduced_basis,
                                       coercivity_estimator=coercivity_estimator,
                                       operator_is_symmetric=operator_symmetric,
                                       dual_bases=dual_reduced_bases)

                # two different roms
                rom_standard = RB_reductor.reduce()
                rom_restricted = RB_reductor.reduce(2)

                for mu in random_set:
                    s_fom = fom.output(mu=mu)
                    for rom in [rom_standard, rom_restricted]:
                        s_rom, s_est = rom.output(return_error_estimate=True, mu=mu)
                        estimator_values.append(s_est)
                        for s_r, s_f, s_e in np.dstack((s_rom, s_fom, s_est))[0]:
                            assert np.abs(s_r-s_f) <= s_e + 1e-12


def create_fom(grid_intervals, vector_valued_output=False):
    p = thermal_block_problem([3, 3])
    f = ConstantFunction(1, dim_domain=2)

    if vector_valued_output:
        p = p.with_(outputs=[('l2', f), ('l2', f * 0.5)])
    else:
        p = p.with_(outputs=[('l2', f)])

    fom, _ = discretize_stationary_cg(p, diameter=1./grid_intervals)
    return fom


if __name__ == '__main__':
    run(main)
