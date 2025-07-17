#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import time

import numpy as np
from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissingError
from pymor.reductors.neural_network import NeuralNetworkReductor, NeuralNetworkStatefreeOutputReductor


def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),
    validation_samples: int = Argument(..., help='Number of samples used for validation during the training phase.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
):
    """Model order reduction with neural networks (approach by Hesthaven and Ubbiali)."""
    if not config.HAVE_TORCH:
        raise TorchMissingError

    fom = create_fom(fv, grid_intervals)

    parameter_space = fom.parameters.space((0.1, 1))

    training_parameters = parameter_space.sample_uniformly(training_samples)
    validation_parameters = parameter_space.sample_randomly(validation_samples)
    test_parameters = parameter_space.sample_randomly(10)

    if getattr(sys, '_called_from_test', False):
        reductor = NeuralNetworkReductor(fom=fom, training_parameters=training_parameters,
                                         validation_parameters=validation_parameters,
                                         l2_err=1e-5, ann_mse=1e-5)
        rom = reductor.reduce(restarts=100, log_loss_frequency=10)

        U_red = fom.solution_space.empty(reserve=len(test_parameters))
        for mu in test_parameters:
            U_red.append(reductor.reconstruct(rom.solve(mu)))

    training_outputs = []
    training_snapshots = fom.solution_space.empty(reserve=len(training_parameters))
    for mu in training_parameters:
        res = fom.compute(solution=True, output=True, mu=mu)
        training_snapshots.append(res['solution'])
        training_outputs.append(res['output'].flatten())
    training_outputs = np.array(training_outputs).T

    validation_outputs = []
    validation_snapshots = fom.solution_space.empty(reserve=len(validation_parameters))
    for mu in validation_parameters:
        res = fom.compute(solution=True, output=True, mu=mu)
        validation_snapshots.append(res['solution'])
        validation_outputs.append(res['output'].flatten())
    validation_outputs = np.array(validation_outputs).T

    reductor_data_driven = NeuralNetworkReductor(training_parameters=training_parameters,
                                                 training_snapshots=training_snapshots,
                                                 validation_parameters=validation_parameters,
                                                 validation_snapshots=validation_snapshots,
                                                 l2_err=1e-5, ann_mse=1e-5)
    rom_data_driven = reductor_data_driven.reduce(restarts=100, log_loss_frequency=10)

    speedups_data_driven = []

    print(f'Performing test on parameters of size {len(test_parameters)} ...')

    U = fom.solution_space.empty(reserve=len(test_parameters))
    U_red_data_driven = fom.solution_space.empty(reserve=len(test_parameters))

    for mu in test_parameters:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        U_red_data_driven.append(reductor_data_driven.reconstruct(rom_data_driven.solve(mu)))
        time_red_data_driven = time.perf_counter() - tic

        speedups_data_driven.append(time_fom / time_red_data_driven)

    absolute_errors_data_driven = (U - U_red_data_driven).norm()
    relative_errors_data_driven = (U - U_red_data_driven).norm() / U.norm()

    if vis:
        fom.visualize((U, U_red_data_driven),
                      legend=('Full solution', 'Reduced solution (data-driven)'))

    output_reductor = NeuralNetworkStatefreeOutputReductor(fom=fom, training_parameters=training_parameters,
                                                           validation_parameters=validation_parameters,
                                                           validation_loss=1e-5)
    output_rom = output_reductor.reduce(restarts=100, log_loss_frequency=10)

    outputs = []
    outputs_red = []
    outputs_speedups = []

    print(f'Performing test on parameters of size {len(test_parameters)} ...')

    for mu in test_parameters:
        tic = time.perf_counter()
        outputs.append(fom.output(mu=mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red.append(output_rom.output(mu=mu))
        time_red = time.perf_counter() - tic

        outputs_speedups.append(time_fom / time_red)

    outputs = np.squeeze(np.array(outputs))
    outputs_red = np.squeeze(np.array(outputs_red))

    outputs_absolute_errors = np.abs(outputs - outputs_red)
    outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)

    # data-driven output reductor
    output_reductor_data_driven = NeuralNetworkStatefreeOutputReductor(training_parameters=training_parameters,
                                                                       training_outputs=training_outputs,
                                                                       validation_parameters=validation_parameters,
                                                                       validation_outputs=validation_outputs,
                                                                       validation_loss=1e-5)
    output_rom_data_driven = output_reductor_data_driven.reduce(restarts=100, log_loss_frequency=10)

    outputs = []
    outputs_red_data_driven = []
    outputs_speedups_data_driven = []
    print(f'Performing test on parameters of size {len(test_parameters)} ...')

    for mu in test_parameters:
        tic = time.perf_counter()
        outputs.append(fom.output(mu=mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red_data_driven.append(output_rom_data_driven.output(mu=mu))
        time_red_data_driven = time.perf_counter() - tic

        outputs_speedups_data_driven.append(time_fom / time_red_data_driven)

    outputs = np.squeeze(np.array(outputs))
    outputs_red_data_driven = np.squeeze(np.array(outputs_red))

    outputs_absolute_errors_data_driven = np.abs(outputs - outputs_red_data_driven)
    outputs_relative_errors_data_driven = np.abs(outputs - outputs_red_data_driven) / np.abs(outputs)

    print()
    print('Results for state approximation purely data-driven:')
    print(f'Average absolute error: {np.average(absolute_errors_data_driven)}')
    print(f'Average relative error: {np.average(relative_errors_data_driven)}')
    print(f'Median of speedup: {np.median(speedups_data_driven)}')

    print()
    print('Results for output approximation with `NeuralNetworkStatefreeOutputReductor`:')
    print(f'Average absolute error: {np.average(outputs_absolute_errors)}')
    print(f'Average relative error: {np.average(outputs_relative_errors)}')
    print(f'Median of speedup: {np.median(outputs_speedups)}')

    print()
    print('Results for data-driven output approximation with `NeuralNetworkStatefreeOutputReductor`:')
    print(f'Average absolute error: {np.average(outputs_absolute_errors_data_driven)}')
    print(f'Average relative error: {np.average(outputs_relative_errors_data_driven)}')
    print(f'Median of speedup: {np.median(outputs_speedups_data_driven)}')

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
