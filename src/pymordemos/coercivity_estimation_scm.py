# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)


import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App

from pymor.algorithms.scm import construct_scm_functionals
from pymor.basic import *

# parameters for high-dimensional models
XBLOCKS = 2             # pyMOR/FEniCS
YBLOCKS = 2             # pyMOR/FEniCS
GRID_INTERVALS = 10     # pyMOR/FEniCS


####################################################################################################
# Main script                                                                                      #
####################################################################################################

TAGS = ('stationary', 'linear', 'parametric', 'error-estimation', 'builtin-discretizer')

app = App(help_on_error=True)

@app.default
def main(
    num_training_parameters: int = 50,
    num_test_parameters: int = 10,
    max_extensions: int = 10,
    num_neighbors: int = 5
):
    """Simplified version of the thermalblock demo to showcase the successive constraints method.

    This demo estimates the coercivity constant of a parametric operator using the successive
    constraints method (SCM, :cite:`HRSP07`). The full-order model is a 2x2 thermal block
    problem (:func:`~pymor.analyticalproblems.thermalblock.thermal_block_problem`) discretized
    with linear finite elements on a coarse grid using
    :func:`~pymor.discretizers.builtin.cg.discretize_stationary_cg`. Using a weak greedy
    algorithm on a random training set, :func:`~pymor.algorithms.scm.construct_scm_functionals`
    selects constraint parameters and constructs parameter functionals that provide lower and
    upper bounds for the coercivity constant with respect to the :math:`H^1_0`-seminorm
    (:class:`~pymor.algorithms.scm.LBSuccessiveConstraintsFunctional` and
    :class:`~pymor.algorithms.scm.UBSuccessiveConstraintsFunctional`). The lower bound is
    obtained by solving a linear program that takes the closest constraint parameters into
    account. The plots show the maximum greedy errors over the extension steps and, for random
    test parameters, the lower and upper bounds together with the true coercivity constant,
    which for the thermal block problem is the minimum of the diffusion coefficients.

    Lower bounds for the coercivity constant are required by the residual-based a posteriori
    error estimators of reduced basis methods. For a reduced basis approximation of the thermal
    block problem, where the coercivity constant is known explicitly, see
    :ref:`cyclopts-pymor-demo-thermalblock`.

    Parameters
    ----------
    num_training_parameters
        Number of training parameters.
    num_test_parameters
        Number of test parameters.
    max_extensions
        Maximum number of extensions of the constraint parameter set.
    num_neighbors
        Number of neighbors in the parameter space used to compute bounds.
    """
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
    app()
