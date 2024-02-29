#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo to showcase the successive constraints method.

Usage:
  coercivity_estimation_scm.py ALG SNAPSHOTS RBSIZE TEST
"""

from typer import Argument, run

from pymor.basic import *
from pymor.parameters.functionals import LBSuccessiveConstraintsFunctional, UBSuccessiveConstraintsFunctional
from pymor.tools.typer import Choices

# parameters for high-dimensional models
XBLOCKS = 2             # pyMOR/FEniCS
YBLOCKS = 2             # pyMOR/FEniCS
GRID_INTERVALS = 10     # pyMOR/FEniCS


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main(
    alg: Choices('naive greedy adaptive_greedy pod') = Argument(..., help='The model reduction algorithm to use.'),
    snapshots: int = Argument(
        ...,
        help='naive: ignored.\n\n'
             'greedy/pod: Number of training_set parameters per block'
             '(in total SNAPSHOTS^(XBLOCKS * YBLOCKS) parameters).\n\n'
             'adaptive_greedy: size of validation set.'
    ),
    rbsize: int = Argument(..., help='Size of the reduced basis.'),
    test: int = Argument(..., help='Number of parameters for stochastic error estimation.'),
):
    # discretize
    ############
    fom, parameter_space = discretize_pymor()

    # select reduction algorithm with error estimator
    #################################################
    bounds = None
    coercivity_constants = None
    num_constraint_parameters = 20
    constraint_parameters = parameter_space.sample_randomly(num_constraint_parameters)
    coercivity_estimator = LBSuccessiveConstraintsFunctional(fom.operator, constraint_parameters, M=5,
                                                             bounds=bounds, coercivity_constants=coercivity_constants)
    upper_coercivity_estimator = UBSuccessiveConstraintsFunctional(fom.operator, constraint_parameters)

    num_test_parameters = 10
    test_parameters = parameter_space.sample_randomly(num_test_parameters)
    print(f'Results for coercivity estimation on a set of {num_test_parameters} randomly chosen test parameters:')
    for mu in test_parameters:
        lb = coercivity_estimator.evaluate(mu)
        ub = upper_coercivity_estimator.evaluate(mu)
        print(f'\tlb: {lb:.5f}\tub: {ub:.5f}\trel.diff.: {2 * (ub-lb) / (lb + ub):.5f}')

    reductor = CoerciveRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator,
                                  check_orthonormality=False)

    # generate reduced model
    ########################
    if alg == 'naive':
        from pymordemos.thermalblock_simple import reduce_naive
        rom = reduce_naive(fom, reductor, parameter_space, rbsize)
    elif alg == 'greedy':
        from pymordemos.thermalblock_simple import reduce_greedy
        rom = reduce_greedy(fom, reductor, parameter_space, snapshots, rbsize)
    elif alg == 'adaptive_greedy':
        from pymordemos.thermalblock_simple import reduce_adaptive_greedy
        rom = reduce_adaptive_greedy(fom, reductor, parameter_space, snapshots, rbsize)
    elif alg == 'pod':
        from pymordemos.thermalblock_simple import reduce_pod
        rom = reduce_pod(fom, reductor, parameter_space, snapshots, rbsize)
    else:
        raise NotImplementedError

    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(rom, fom=fom, reductor=reductor, error_estimator=True,
                                       error_norms=[fom.h1_0_semi_norm], condition=True,
                                       test_mus=parameter_space.sample_randomly(test))

    # show results
    ##############
    print(results['summary'])
    plot_reduction_error_analysis(results)

    # write results to disk
    #######################
    from pymor.core.pickle import dump
    with open('reduced_model.out', 'wb') as f:
        dump((rom, parameter_space), f)
    with open('results.out', 'wb') as f:
        dump(results, f)

    # visualize reduction error for worst-approximated mu
    #####################################################
    mumax = results['max_error_mus'][0, -1]
    U = fom.solve(mumax)
    U_RB = reductor.reconstruct(rom.solve(mumax))
    fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                  separate_colorbars=True, block=True)


####################################################################################################
# High-dimensional model                                                                           #
####################################################################################################


def discretize_pymor():

    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(XBLOCKS, YBLOCKS))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / GRID_INTERVALS)

    return fom, problem.parameter_space


if __name__ == '__main__':
    run(main)
