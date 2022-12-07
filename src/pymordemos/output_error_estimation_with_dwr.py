# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import numpy as np
import matplotlib.pyplot as plt
from typer import Argument, run

from pymor.basic import *
from pymor.reductors.dwr import DWRCoerciveRBReductor


def main(
    fom_number: int = Argument(..., help='Selects FOMs [0, 1] for elliptic problems '
                                         + 'with scalar and vector valued outputs '),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the reduced basis.'),
    modes: int = Argument(..., help='Number of basis functions for the RB spaces (generated with POD)')
):
    """Demo script for using DWR-based output error estimation."""
    set_log_levels({'pymor': 'INFO'})

    assert fom_number in [0, 1], f'No FOM available for fom_number {fom_number}'

    # elliptic case
    if fom_number == 0:
        # real valued output
        fom = create_fom(grid_intervals, vector_valued_output=False)
    elif fom_number == 1:
        # vector valued output (with BlockColumnOperator)
        fom = create_fom(grid_intervals, vector_valued_output=True)

    # Parameter space and operator are equal for all fom
    parameter_space = fom.parameters.space(0.1, 1)
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)
    training_set = parameter_space.sample_uniformly(training_samples)

    # generate solution snapshots
    primal_snapshots = fom.solution_space.empty()
    # store true outputs
    fom_outputs = []

    # construct training data
    for mu in training_set:
        comp_data = fom.compute(output=True, solution=True, mu=mu)
        primal_snapshots.append(comp_data['solution'])
        fom_outputs.append(comp_data['output'])

    # apply POD on bases
    product = fom.h1_0_product
    primal_RB, _ = pod(primal_snapshots, modes=modes, product=product)

    # standard RB reductor for comparison
    standard_RB_reductor = CoerciveRBReductor(fom, RB=primal_RB, product=product,
                                              coercivity_estimator=coercivity_estimator)

    # initialize DWR reductor
    dwr_RB_reductor = DWRCoerciveRBReductor(fom, dual_foms=None, product=product,
                                            coercivity_estimator=coercivity_estimator)

    # also construct dual bases for dwr
    dual_RBs = []
    dual_foms = dwr_RB_reductor.dual_foms
    for d in range(fom.dim_output):
        dual_snapshots = fom.solution_space.empty()
        # initialize dual model from reductor
        dual_fom = dual_foms[d]
        for mu in training_set:
            dual_snapshots.append(dual_fom.solve(mu))
        # use one mode more to test the case where the size is not the same
        dual_RBs.append(pod(dual_snapshots, modes=modes+1, product=product)[0])

    # extend basis
    dwr_RB_reductor.extend_basis(primal_RB, dual_RBs)

    # dwr reductor without dual basis
    dwr_reductor_primal = DWRCoerciveRBReductor(fom, primal_RB=primal_RB, product=product,
                                                coercivity_estimator=coercivity_estimator)

    dwr_reductors = [dwr_RB_reductor, dwr_reductor_primal]

    # rom
    standard_rom = standard_RB_reductor.reduce()
    roms = [standard_rom]
    roms.extend([dwr_red.reduce() for dwr_red in dwr_reductors])

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
            assert np.linalg.norm(np.abs(s_rom-s_fom)) <= s_est

        results.append(results_full)

    # plot result
    plt.figure()
    plt.semilogy(np.arange(len(training_set)), results[0]['err'], 'ko-', markersize=4,
                 label=f'standard output error basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results[0]['est'], 'k--', alpha=.5,
                 label=f'standard output estimate basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results[1]['err'], 'ro-', alpha=.5,
                 label=f'dwr output error basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results[1]['est'], 'r--', alpha=.5,
                 label=f'dwr output estimate basis size {modes}')
    plt.semilogy(np.arange(len(training_set)), results[2]['err'], 'yo-', alpha=.5,
                 label=f'dwr output error basis size {modes}, no dual_bases')
    plt.semilogy(np.arange(len(training_set)), results[2]['est'], 'y--', alpha=.5,
                 label=f'dwr output estimate basis size {modes}, no dual_bases')
    plt.title(f'Error and estimate for {modes} basis functions for parameters in training set')
    plt.xlim(10, 50)
    plt.legend()

    # estimator study for smaller number of basis functions
    modes_set = np.arange(0, len(primal_RB)+1)
    max_errss, max_estss, min_errss, min_estss = [], [], [], []
    for reductor in [standard_RB_reductor, dwr_reductors[0], dwr_reductors[1]]:
        max_errs, max_ests, min_errs, min_ests = [], [], [], []
        for mode in modes_set:
            max_err, max_est, min_err, min_est = 0, 0, np.inf, np.inf
            if isinstance(reductor, DWRCoerciveRBReductor):
                rom = reductor.reduce(mode, mode)
            else:
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
    plt.semilogy(modes_set, min_errss[0], 'g-o', label='standard min error')
    plt.semilogy(modes_set, min_estss[0], 'g--', label='standard min estimate')
    plt.semilogy(modes_set, min_errss[1], 'm-o', label='DWR min error')
    plt.semilogy(modes_set, min_estss[1], 'm--', label='DWR min estimate')
    plt.legend()
    plt.title('Evolution of minimum error and estimate for different RB sizes')

    plt.figure()
    plt.semilogy(modes_set, max_errss[0], 'g-o', label='standard max error')
    plt.semilogy(modes_set, max_estss[0], 'g--', label='standard max estimate')
    plt.semilogy(modes_set, max_errss[1], 'm-o', label='DWR max error')
    plt.semilogy(modes_set, max_estss[1], 'm--', label='DWR max estimate')
    plt.legend()
    plt.title('Evolution of maximum error and estimate for different RB sizes')

    plt.show()


def create_fom(grid_intervals, vector_valued_output=False):
    p = thermal_block_problem([2, 2])
    f_1 = ConstantFunction(1, dim_domain=2)
    f_2 = ExpressionFunction('sin(x[0])', dim_domain=2)

    if vector_valued_output:
        p = p.with_(outputs=[('l2', f_1), ('l2', f_2 * 0.5)])
    else:
        p = p.with_(outputs=[('l2', f_1)])

    fom, _ = discretize_stationary_cg(p, diameter=1./grid_intervals)
    return fom


if __name__ == '__main__':
    run(main)
