#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.basic import *


def main(
    fom_number: int = Argument(..., help='Selects FOMs [0, 1, 2] for elliptic problems and [3, 4] for '
                                         + 'parabolic problems with scalar and vector valued outputs '),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the reduced basis.'),
    modes: int = Argument(..., help='Number of basis functions for the RB space (generated with POD)'),
    reductor_count: int = Argument(..., help='Reductor type for elliptic problems: \
                                   0: SimpleCoerciveReductor \
                                   1: CoerciveRBReductor. \
                                   For parabolic FOMs [4, 5] ParabolicRBReductor is used.')):
    set_log_levels({'pymor': 'WARN'})
    """Example script for using output error estimation"""

    assert fom_number in [0, 1, 2, 3, 4], f'No FOM available for fom_number {fom_number}'

    # elliptic case
    if fom_number == 0:
        # real valued output
        fom = create_fom(grid_intervals, vector_valued_output=False)
    elif fom_number == 1:
        # vector valued output (with BlockColumnOperator)
        fom = create_fom(grid_intervals, vector_valued_output=True)
    elif fom_number == 2:
        # an output which is actually a lincomb operator
        fom = create_fom(grid_intervals, vector_valued_output=True)
        dim_source = fom.output_functional.source.dim
        np.random.seed(1)
        random_matrix_1 = np.random.rand(2, dim_source)
        random_matrix_2 = np.random.rand(2, dim_source)
        op1 = NumpyMatrixOperator(random_matrix_1, source_id='STATE')
        op2 = NumpyMatrixOperator(random_matrix_2, source_id='STATE')
        ops = [op1, op2]
        lincomb_op = LincombOperator(ops, [1., 0.5])
        fom = fom.with_(output_functional=lincomb_op)
    # parabolic case
    elif fom_number in [3, 4]:
        from pymordemos.parabolic_mor import discretize_pymor
        fom = discretize_pymor()
        if fom_number == 3:
            fom = fom.with_(output_functional=fom.rhs.operators[0].H)
        else:
            random_matrix_1 = np.random.rand(2, fom.solution_space.dim)
            op = NumpyMatrixOperator(random_matrix_1, source_id='STATE')
            fom = fom.with_(output_functional=op)

    if reductor_count == 0:
        reductor = SimpleCoerciveRBReductor
    elif reductor_count == 1:
        reductor = CoerciveRBReductor
    if fom_number in [3, 4]:
        reductor = ParabolicRBReductor

    # Parameter space and operator are equal for all elliptic and parabolic foms
    if fom_number in [0, 1, 2]:
        parameter_space = fom.parameters.space(0.1, 1)
        coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)
    else:
        parameter_space = fom.parameters.space(1, 100)
        coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)

    training_set = parameter_space.sample_uniformly(training_samples)

    # generate solution snapshots
    primal_snapshots = fom.solution_space.empty()
    fom_outputs = []

    # construct training data
    for mu in training_set:
        comp_data = fom.compute(output=True, solution=True, mu=mu)
        primal_snapshots.append(comp_data['solution'])
        fom_outputs.append(comp_data['output'])

    # apply POD on bases
    product = fom.h1_0_semi_product
    primal_reduced_basis, _ = pod(primal_snapshots, modes=modes, product=product)

    RB_reductor = reductor(fom, RB=primal_reduced_basis, product=product,
                           coercivity_estimator=coercivity_estimator)

    # rom
    rom = RB_reductor.reduce()

    results_full = {'fom': [], 'rom': [], 'err': [], 'est': []}
    for i, mu in enumerate(training_set):
        s_fom = fom_outputs[i]
        s_rom, s_est = rom.output(return_error_estimate=True, mu=mu,
                                  return_error_estimate_vector=False)
        results_full['fom'].append(s_fom)
        results_full['rom'].append(s_rom)
        results_full['err'].append(np.linalg.norm(np.abs(s_fom[-1]-s_rom[-1])))
        results_full['est'].append(s_est)

        # just for testing purpose
        assert np.linalg.norm(np.abs(s_rom-s_fom)) <= s_est + 1e-8

        # also test return_estimate_vector and return_error_sequence functionality but do not use it
        s_rom_, s_est_ = rom.output(return_error_estimate=True, mu=mu,
                                    return_error_estimate_vector=True)
        assert np.allclose(s_rom, s_rom_)
        assert np.allclose(s_est, np.linalg.norm(s_est_))

        if fom_number in [3, 4]:
            s_rom__, s_est__ = rom.output(return_error_estimate=True, mu=mu, return_error_sequence=True)
            s_rom___, s_est___ = rom.output(return_error_estimate=True, mu=mu,
                                            return_error_estimate_vector=True,
                                            return_error_sequence=True)
            # s_rom always stays the same
            assert np.allclose(s_rom, s_rom__, s_rom___)
            assert s_est__[-1] == s_est
            assert np.allclose(s_est__, np.linalg.norm(s_est___, axis=1))

    # plot result
    plt.figure()
    plt.semilogy(np.arange(len(training_set)), results_full['err'], 'k', label=f'output error basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results_full['est'], 'k--', label=f'output estimate basis size {modes}')
    plt.title(f'Error and estimate for {modes} basis functions for parameters in training set')
    plt.legend()

    # estimator study for smaller number of basis functions
    modes_set = np.arange(1, rom.solution_space.dim+1)
    max_errs, max_ests, min_errs, min_ests = [], [], [], []
    for mode in modes_set:
        max_err, max_est, min_err, min_est = 0, 0, 1000, 1000
        rom = RB_reductor.reduce(mode)

        for i, mu in enumerate(training_set):
            s_fom = fom_outputs[i]
            s_rom, s_est = rom.output(return_error_estimate=True, mu=mu)
            max_err = max(max_err, np.linalg.norm(np.abs(s_fom-s_rom)))
            max_est = max(max_est, s_est)
            min_err = min(min_err, np.linalg.norm(np.abs(s_fom-s_rom)))
            min_est = min(min_est, s_est)

        max_errs.append(max_err)
        max_ests.append(max_est)
        min_errs.append(min_err)
        min_ests.append(min_est)

    plt.figure()
    plt.semilogy(modes_set, max_errs, 'k', label='max error')
    plt.semilogy(modes_set, max_ests, 'k--', label='max estimate')
    plt.semilogy(modes_set, min_errs, 'g', label='min error')
    plt.semilogy(modes_set, min_ests, 'g--', label='min estimate')
    plt.legend()
    plt.title('Evolution of maximum error and estimate for different RB sizes')
    plt.show()


def create_fom(grid_intervals, vector_valued_output=False):
    p = thermal_block_problem([2, 2])
    f = ConstantFunction(1, dim_domain=2)

    if vector_valued_output:
        p = p.with_(outputs=[('l2', f), ('l2', f * 0.5)])
    else:
        p = p.with_(outputs=[('l2', f)])

    fom, _ = discretize_stationary_cg(p, diameter=1./grid_intervals)
    return fom


if __name__ == '__main__':
    run(main)
