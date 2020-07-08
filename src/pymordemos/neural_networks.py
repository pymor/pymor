# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for the usage of neural networks in model order reduction (approach by Hesthaven and Ubbiali)

Usage:
    neural_networks.py [--fv] [--vis] GRID_INTERVALS TRAINING_SAMPLES VALIDATION_SAMPLES

Arguments:
    GRID_INTERVALS       Grid interval count.
    TRAINING_SAMPLES     Number of samples used for training the neural network.
    VALIDATION_SAMPLES   Number of samples used for validation during the training phase.

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
    --vis        Visualize full order solution and reduced solution for a test set.
"""

from docopt import docopt

import numpy as np

from pymor.basic import *

from pymor.core.config import config
from pymor.core.exceptions import TorchMissing


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
    fom, _ = discretizer(problem, diameter=1. / int(args['GRID_INTERVALS']))

    return fom


def neural_networks_demo(args):
    logger = getLogger('pymordemos.neural_networks')

    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom = create_fom(args)

    parameter_space = fom.parameters.space((0.1, 1))

    from pymor.reductors.neural_network import NeuralNetworkReductor

    training_set = parameter_space.sample_uniformly(int(args['TRAINING_SAMPLES']))
    validation_set = parameter_space.sample_randomly(int(args['VALIDATION_SAMPLES']))

    reductor = NeuralNetworkReductor(fom, training_set, validation_set, l2_err=1e-5,
                                     ann_mse=1e-5)
    rom = reductor.reduce()

    test_set = parameter_space.sample_randomly(10)

    speedups = []

    import time

    print(f'Performing test on set of size {len(test_set)} ...')

    U = fom.solution_space.empty(reserve=len(test_set))
    U_red = fom.solution_space.empty(reserve=len(test_set))

    for mu in test_set:
        tic = time.time()
        U.append(fom.solve(mu))
        time_fom = time.time() - tic

        tic = time.time()
        U_red.append(reductor.reconstruct(rom.solve(mu)))
        time_red = time.time() - tic

        speedups.append(time_fom / time_red)

    absolute_errors = (U - U_red).l2_norm()
    relative_errors = (U - U_red).l2_norm() / U.l2_norm()

    if args['--vis']:
        fom.visualize((U, U_red),
                      legend=('Full solution', 'Reduced solution'))

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    neural_networks_demo(args)
