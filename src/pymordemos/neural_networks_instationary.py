# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Example script for the usage of neural networks in model order reduction for an instationary problem (approach by Hesthaven and Ubbiali)

Usage:
    neural_networks_instationary.py [--fv] GRID_INTERVALS TIME_STEPS TRAINING_SAMPLES VALIDATION_SAMPLES

Arguments:
    GRID_INTERVALS       Grid interval count.
    TIME_STEPS           Number of time steps used for discretization.
    TRAINING_SAMPLES     Number of samples used for training the neural network.
    VALIDATION_SAMPLES   Number of samples used for validation during the training phase.

Options:
    -h, --help   Show this message.
    --fv         Use finite volume discretization instead of finite elements.
"""

from docopt import docopt

import numpy as np

from pymor.basic import *

from pymor.core.config import config
from pymor.core.exceptions import TorchMissing


def create_fom(args):
    problem = burgers_problem()

    print('Discretize ...')
    discretizer = discretize_instationary_fv
    fom, _ = discretizer(problem, diameter=1. / int(args['GRID_INTERVALS']), nt=int(args['TIME_STEPS']))

    return fom


def neural_networks_instationary_demo(args):
    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom = create_fom(args)

    Nt = int(args['TIME_STEPS'])

    parameter_space = fom.parameters.space(1., 2.)

    from pymor.reductors.neural_network import NeuralNetworkInstationaryReductor

    training_set = parameter_space.sample_uniformly(int(args['TRAINING_SAMPLES']))
    validation_set = parameter_space.sample_randomly(int(args['VALIDATION_SAMPLES']))

    reductor = NeuralNetworkInstationaryReductor(fom, Nt, training_set, validation_set, basis_size=10)
    rom = reductor.reduce(hidden_layers='[30, 30, 30]', restarts=100)

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

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


if __name__ == '__main__':
    args = docopt(__doc__)
    neural_networks_instationary_demo(args)
