# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for the usage of neural networks in model order reduction (approach by Hesthaven and Ubbiali)

Usage:
    neural_networks.py [--fv] [--vis] N TRAINING_SAMPLES VALIDATION_SAMPLES

Arguments:
    N                Grid interval count

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
    --vis        Visualize full order solution and reduced solution for a test set.
"""

from docopt import docopt

import numpy as np

from pymor.analyticalproblems.domaindescriptions import LineDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ExpressionFunction, ConstantFunction, LincombFunction
from pymor.core.config import config
from pymor.core.logger import getLogger
from pymor.discretizers.builtin import discretize_stationary_cg, discretize_stationary_fv
from pymor.parameters.functionals import ProjectionParameterFunctional


def create_fom(args):
    rhs = ExpressionFunction('(x - 0.5)**2 * 1000', 1, ())

    d0 = ExpressionFunction('1 - x', 1, ())
    d1 = ExpressionFunction('x', 1, ())

    f0 = ProjectionParameterFunctional('diffusionl')
    f1 = 1.

    problem = StationaryProblem(
        domain=LineDomain(),
        rhs=rhs,
        diffusion=LincombFunction([d0, d1], [f0, f1]),
        dirichlet_data=ConstantFunction(value=0, dim_domain=1),
        name='1DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if args['--fv'] else discretize_stationary_cg
    m, _ = discretizer(problem, diameter=1. / int(args['N']))

    return m


def neural_networks_demo(args):
    logger = getLogger('pymordemos.neural_networks')

    if not config.HAVE_TORCH:
        logger.error('PyTorch is not installed! Stopping.')
        return

    TRAINING_SAMPLES = args['TRAINING_SAMPLES']
    VALIDATION_SAMPLES = args['VALIDATION_SAMPLES']

    fom = create_fom(args)

    parameter_space = fom.parameters.space((0.1, 1))

    from pymor.reductors.neural_network import NeuralNetworkReductor

    training_set = parameter_space.sample_uniformly(int(TRAINING_SAMPLES))
    validation_set = parameter_space.sample_randomly(int(VALIDATION_SAMPLES))

    basis_size = 10

    reductor = NeuralNetworkReductor(fom, training_set, validation_set, basis_size=basis_size)
    rom = reductor.reduce()

    test_set = parameter_space.sample_randomly(10)

    absolute_errors = []
    relative_errors = []
    speedups = []

    import time

    print(f'Performing test on set of size {len(test_set)} ...')

    for mu in test_set:
        tic = time.time()
        U = fom.solve(mu)
        time_fom = time.time() - tic

        tic = time.time()
        U_red = rom.solve(mu)
        time_red = time.time() - tic

        if args['--vis']:
            fom.visualize(U, title=f'Full solution for mu={mu}')
            fom.visualize(U_red, title=f'Reduced solution for mu={mu}')

        absolute_errors.append((U - U_red).l2_norm())
        relative_errors.append((U - U_red).l2_norm() / U.l2_norm())
        speedups.append(time_fom / time_red)


    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    neural_networks_demo(args)
