#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.basic import *
from pymor.reductors.dwr import DWRCoerciveRBReductor


def main(
    fom_number: int = Argument(..., help='Selects FOMs [0, 1, 2] for elliptic problems '
                                         + 'with scalar and vector valued outputs '),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the reduced basis.'),
    modes: int = Argument(..., help='Number of basis functions for the RB spaces (generated with POD)')
):
    set_log_levels({'pymor': 'WARN'})
    """Example script for using output error estimation compared with dwr approach"""

    assert fom_number in [0, 1, 2], f'No FOM available for fom_number {fom_number}'

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

    standard_reductor = CoerciveRBReductor
    dwr_reductor = DWRCoerciveRBReductor

    # Parameter space and operator are equal for all fom
    parameter_space = fom.parameters.space(0.1, 1)
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)
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

    standard_RB_reductor = standard_reductor(fom, RB=primal_reduced_basis, product=product,
                                             coercivity_estimator=coercivity_estimator)

    # also construct dual bases for dwr
    # take the operator as the dual operator if it is symmetric
    symmetries = [True, False]
    dwr_reductors = []
    for operator_symmetric in symmetries:
        dual_reduced_bases = []
        for d in range(fom.dim_output):
            dual_snapshots = fom.solution_space.empty()
            # initialize dual model from reductor
            dual_fom = dwr_reductor.dual_model(fom, d, operator_symmetric)
            for mu in training_set:
                dual_snapshots.append(dual_fom.solve(mu))
            dual_reduced_bases.append(pod(dual_snapshots, modes=modes, product=product)[0])

        dwr_RB_reductor = dwr_reductor(fom,
                                       primal_basis=primal_reduced_basis,
                                       product=product,
                                       dual_bases=dual_reduced_bases,
                                       coercivity_estimator=coercivity_estimator,
                                       operator_is_symmetric=operator_symmetric)
        dwr_reductors.append(dwr_RB_reductor)

    # dwr reductor without dual basis
    dwr_reductor_primal = dwr_reductor(fom,
                                       primal_basis=primal_reduced_basis,
                                       product=product,
                                       coercivity_estimator=coercivity_estimator,
                                       operator_is_symmetric=operator_symmetric)
    dwr_reductors.append(dwr_reductor_primal)

    # rom
    standard_rom = standard_RB_reductor.reduce()
    dwr_roms = [dwr_red.reduce() for dwr_red in dwr_reductors]
    roms = [standard_rom, dwr_roms[0], dwr_roms[1], dwr_roms[2]]

    results = []
    for rom in roms:
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
            assert np.linalg.norm(np.abs(s_rom-s_fom)) <= s_est + 1e-7
        results.append(results_full)

    # plot result
    plt.figure()
    plt.semilogy(np.arange(len(training_set)), results[0]['err'], 'k',
                 label=f'standard output error basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results[0]['est'], 'k--',
                 label=f'standard output estimate basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results[1]['err'], 'g',
                 label=f'dwr output error basis size {modes}, operator_symmetric=True')
    plt.semilogy(np.arange(len(training_set)), results[1]['est'], 'g--',
                 label=f'dwr output estimate basis size {modes}, operator_symmetric=True')
    plt.semilogy(np.arange(len(training_set)), results[2]['err'], 'r',
                 label=f'dwr output error basis size {modes}, operator_symmetric=False')
    plt.semilogy(np.arange(len(training_set)), results[2]['est'], 'r--',
                 label=f'dwr output estimate basis size {modes}, operator_symmetric=False')
    plt.semilogy(np.arange(len(training_set)), results[3]['err'], 'y',
                 label=f'dwr output error basis size {modes}, no dual_bases')
    plt.semilogy(np.arange(len(training_set)), results[3]['est'], 'y--',
                 label=f'dwr output estimate basis size {modes}, no dual_basis')
    plt.title(f'Error and estimate for {modes} basis functions for parameters in training set')
    plt.legend()
    # plt.show()

    # estimator study for smaller number of basis functions
    modes_set = np.arange(1, len(primal_reduced_basis)+1)
    max_errss, max_estss, min_errss, min_estss = [], [], [], []
    for reductor in [standard_RB_reductor, dwr_reductors[0], dwr_reductors[1]]:
        max_errs, max_ests, min_errs, min_ests = [], [], [], []
        for mode in modes_set:
            max_err, max_est, min_err, min_est = 0, 0, 1000, 1000
            rom = reductor.reduce(mode)

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

        max_errss.append(max_errs)
        max_estss.append(max_ests)
        min_errss.append(min_errs)
        min_estss.append(min_ests)

    plt.figure()
    # plt.semilogy(modes_set, max_errss[0], 'k', label='standard max error')
    # plt.semilogy(modes_set, max_estss[0], 'k--', label='standard max estimate')
    plt.semilogy(modes_set, min_errss[0], 'g', label='standard min error')
    plt.semilogy(modes_set, min_estss[0], 'g--', label='standard min estimate')
    # plt.semilogy(modes_set, max_errss[1], 'r', label='dwr max error, 1')
    # plt.semilogy(modes_set, max_estss[1], 'r--', label='dwr max estimate, 1')
    plt.semilogy(modes_set, min_errss[1], 'b',
                 label='dwr min error, operator_symmetric=True')
    plt.semilogy(modes_set, min_estss[1], 'b--',
                 label='dwr min estimate, operator_symmetric=True')
    # plt.semilogy(modes_set, max_errss[2], 'y', label='dwr max error, 2')
    # plt.semilogy(modes_set, max_estss[2], 'y--', label='dwr max estimate, 2')
    plt.semilogy(modes_set, min_errss[2], 'm',
                 label='dwr min error, operator_symmetric=False')
    plt.semilogy(modes_set, min_estss[2], 'm--',
                 label='dwr min estimate, operator_symmetric=False')
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
