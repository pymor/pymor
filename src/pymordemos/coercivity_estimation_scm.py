#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo to showcase the successive constraints method.

Usage:
  coercivity_estimation_scm.py ALG SNAPSHOTS RBSIZE TEST
"""

import matplotlib.pyplot as plt
import numpy as np
from typer import Option, run

from pymor.algorithms.scm import construct_scm_functionals
from pymor.basic import *

# parameters for high-dimensional models
XBLOCKS = 2             # pyMOR/FEniCS
YBLOCKS = 2             # pyMOR/FEniCS
GRID_INTERVALS = 20     # pyMOR/FEniCS


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main(
    num_test_parameters: int = Option(100, help='Number of test parameters.'),
    num_neighbors: int = Option(10, help='Number of neighbors in the parameter space used to compute bounds.')
):
    problem = thermal_block_problem(num_blocks=(XBLOCKS, YBLOCKS))

    fom, _ = discretize_stationary_cg(problem, diameter=1. / GRID_INTERVALS)

    test_parameters = problem.parameter_space.sample_randomly(num_test_parameters)
    list_num_constraint_parameters = np.arange(10, 100, 20)
    results = []
    for num_constraint_parameters in list_num_constraint_parameters:
        constraint_parameters = problem.parameter_space.sample_randomly(num_constraint_parameters)
        coercivity_estimator, upper_coercivity_estimator = construct_scm_functionals(fom.operator,
                                                                                     constraint_parameters,
                                                                                     M=num_neighbors,
                                                                                     product=fom.h1_0_semi_product)

        results_temp = []
        for mu in test_parameters:
            lb = coercivity_estimator.evaluate(mu)
            ub = upper_coercivity_estimator.evaluate(mu)
            true_coercivity_constant = np.min(mu.to_numpy())
            results_temp.append([true_coercivity_constant, lb, ub])
        results.append(results_temp)

    results = np.array(results)

    plt.plot(list_num_constraint_parameters, np.mean(results[..., 0], axis=-1), label='True coercivity constant')
    plt.plot(list_num_constraint_parameters, np.mean(results[..., 1], axis=-1), label='Lower bound')
    plt.plot(list_num_constraint_parameters, np.mean(results[..., 2], axis=-1), label='Upper bound')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(main)
