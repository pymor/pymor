#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
import numpy as np
from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissing
from pymor.reductors.neural_network import NeuralNetworkReductor, NeuralNetworkStatefreeOutputReductor


def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),
    validation_samples: int = Argument(..., help='Number of samples used for validation during the training phase.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
):
    """Model oder reduction with neural networks (approach by Hesthaven and Ubbiali)."""
    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom = create_fom(fv, grid_intervals)

    parameter_space = fom.parameters.space((0.1, 1))

    training_set = parameter_space.sample_uniformly(training_samples)
    validation_set = parameter_space.sample_randomly(validation_samples)

    reductor = NeuralNetworkReductor(fom, training_set, validation_set, l2_err=1e-5,
                                     ann_mse=1e-5)
    rom = reductor.reduce(restarts=100)

    test_set = parameter_space.sample_randomly(10)

    speedups = []

    print(f'Performing test on set of size {len(test_set)} ...')

    U = fom.solution_space.empty(reserve=len(test_set))
    U_red = fom.solution_space.empty(reserve=len(test_set))

    for mu in test_set:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        U_red.append(reductor.reconstruct(rom.solve(mu)))
        time_red = time.perf_counter() - tic

        speedups.append(time_fom / time_red)

    absolute_errors = (U - U_red).norm()
    relative_errors = (U - U_red).norm() / U.norm()

    if vis:
        fom.visualize((U, U_red),
                      legend=('Full solution', 'Reduced solution'))

    output_reductor = NeuralNetworkStatefreeOutputReductor(fom, training_set, validation_set, validation_loss=1e-5)
    output_rom = output_reductor.reduce(restarts=100)

    outputs = []
    outputs_red = []
    outputs_speedups = []
    outputs_red_ann_reductor = []
    outputs_speedups_ann_reductor = []

    print(f'Performing test on set of size {len(test_set)} ...')

    for mu in test_set:
        tic = time.perf_counter()
        outputs.append(fom.compute(output=True, mu=mu)['output'])
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red.append(output_rom.compute(output=True, mu=mu)['output'])
        time_red = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red_ann_reductor.append(rom.compute(output=True, mu=mu)['output'])
        time_red_ann_reductor = time.perf_counter() - tic

        outputs_speedups.append(time_fom / time_red)
        outputs_speedups_ann_reductor.append(time_fom / time_red_ann_reductor)

    outputs = np.squeeze(np.array(outputs))
    outputs_red = np.squeeze(np.array(outputs_red))
    outputs_red_ann_reductor = np.squeeze(np.array(outputs_red_ann_reductor))

    outputs_absolute_errors = np.abs(outputs - outputs_red)
    outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)

    outputs_absolute_errors_ann_reductor = np.abs(outputs - outputs_red_ann_reductor)
    outputs_relative_errors_ann_reductor = np.abs(outputs - outputs_red_ann_reductor) / np.abs(outputs)

    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')

    print()
    print('Results for output approximation with `NeuralNetworkReductor`:')
    print(f'Average absolute error: {np.average(outputs_absolute_errors_ann_reductor)}')
    print(f'Average relative error: {np.average(outputs_relative_errors_ann_reductor)}')
    print(f'Median of speedup: {np.median(outputs_speedups_ann_reductor)}')

    print()
    print('Results for output approximation with `NeuralNetworkStatefreeOutputReductor`:')
    print(f'Average absolute error: {np.average(outputs_absolute_errors)}')
    print(f'Average relative error: {np.average(outputs_relative_errors)}')
    print(f'Median of speedup: {np.median(outputs_speedups)}')


def create_fom(fv, grid_intervals):
    f = LincombFunction(
        [ExpressionFunction('10', 2), ConstantFunction(1., 2)],
        [ProjectionParameterFunctional('mu'), 0.1])
    g = LincombFunction(
        [ExpressionFunction('2 * x[0]', 2), ConstantFunction(1., 2)],
        [ProjectionParameterFunctional('mu'), 0.5])

    problem = StationaryProblem(
        domain=RectDomain(),
        rhs=f,
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[0]', 2), ExpressionFunction('x[0]', 2)],
            [ProjectionParameterFunctional('mu'), 1]),
        dirichlet_data=g,
        outputs=[('l2', f), ('l2_boundary', g)],
        name='2DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
    fom, _ = discretizer(problem, diameter=1. / int(grid_intervals))

    return fom


if __name__ == '__main__':
    run(main)
