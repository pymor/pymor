# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import matplotlib.pyplot as plt
import numpy as np
from typer import Option, run

from pymor.algorithms.scm import construct_scm_functionals
from pymor.basic import *

# parameters for high-dimensional models
XBLOCKS = 2             # pyMOR/FEniCS
YBLOCKS = 2             # pyMOR/FEniCS
GRID_INTERVALS = 10     # pyMOR/FEniCS


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main(
    num_training_parameters: int = Option(50, help='Number of test parameters.'),
    num_test_parameters: int = Option(10, help='Number of test parameters.'),
    max_extensions: int = Option(10, help='Maximum number of extensions of the constraint parameter set.'),
    num_neighbors: int = Option(5, help='Number of neighbors in the parameter space used to compute bounds.')
):
    """Simplified version of the thermalblock demo to showcase the successive constraints method."""
    problem = thermal_block_problem(num_blocks=(XBLOCKS, YBLOCKS))

    fom, _ = discretize_stationary_cg(problem, diameter=1. / GRID_INTERVALS)

    test_parameters = problem.parameter_space.sample_randomly(num_test_parameters)
    initial_parameter = problem.parameter_space.sample_randomly(1)[0]
    training_set = problem.parameter_space.sample_randomly(num_training_parameters)
    coercivity_estimator, upper_coercivity_estimator, greedy_results = construct_scm_functionals(
            fom.operator, training_set, initial_parameter, max_extensions=max_extensions,
            product=fom.h1_0_semi_product, M=num_neighbors)

    results = []
    for mu in test_parameters:
        lb = coercivity_estimator.evaluate(mu)
        ub = upper_coercivity_estimator.evaluate(mu)
        true_coercivity_constant = np.min(mu.to_numpy())
        results.append([true_coercivity_constant, lb, ub])

    results = np.array(results)

    plt.plot(np.arange(len(greedy_results['max_errs'])), greedy_results['max_errs'], label='Maximum greedy errors')
    plt.legend()
    plt.show()

    plt.plot(np.arange(len(test_parameters)), results[..., 0], 'x', label='True coercivity constant')
    plt.plot(np.arange(len(test_parameters)), results[..., 1], label='Lower bound')
    plt.plot(np.arange(len(test_parameters)), results[..., 2], label='Upper bound')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run(main)
