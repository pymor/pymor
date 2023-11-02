#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.reductors.neural_network import (
    NeuralNetworkInstationaryReductor,
    NeuralNetworkInstationaryStatefreeOutputReductor,
    NeuralNetworkLSTMInstationaryReductor,
    NeuralNetworkLSTMInstationaryStatefreeOutputReductor,
)


def main(
    problem_number: int = Argument(..., min=0, max=1, help='Selects the problem to solve [0 or 1].'),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    time_steps: int = Argument(..., help='Number of time steps used for discretization.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),
    validation_samples: int = Argument(..., help='Number of samples used for validation during the training phase.'),
    plot_test_solutions: bool = Option(False, help='Plot FOM and ROM solutions in the test set.'),
):
    """Model oder reduction with neural networks for instationary problems.

    Problem number 0 considers the incompressible Navier-Stokes equations in
    a two-dimensional cavity with the Reynolds number as parameter.
    The discretization is based on FEniCS.

    Problem number 1 considers a parametrized Burgers equation on a
    one-dimensional domain. The discretization is based on pyMOR's built-in
    functionality.
    """
    config.require('TORCH')

    fom, plot_function = create_fom(problem_number, grid_intervals, time_steps)

    if problem_number == 0:
        parameter_space = fom.parameters.space(1., 50.)
    else:
        parameter_space = fom.parameters.space(1., 2.)

    training_set = parameter_space.sample_uniformly(training_samples)
    validation_set = parameter_space.sample_randomly(validation_samples)
    test_set = parameter_space.sample_randomly(10)

    def compute_errors_state(rom, reductor):
        speedups = []

        print(f'Performing test on set of size {len(test_set)} ...')

        U = fom.solution_space.empty(reserve=len(test_set))
        U_red = fom.solution_space.empty(reserve=len(test_set))

        for mu in test_set:
            tic = time.time()
            u_fom = fom.solve(mu)[1:]
            U.append(u_fom)
            time_fom = time.time() - tic
            if plot_test_solutions and plot_function:
                plot_function(u_fom, title='FOM')

            tic = time.time()
            u_red = reductor.reconstruct(rom.solve(mu))[1:]
            U_red.append(u_red)
            time_red = time.time() - tic
            if plot_test_solutions and plot_function:
                plot_function(u_red, title='ROM')

            speedups.append(time_fom / time_red)

        absolute_errors = (U - U_red).norm2()
        relative_errors = (U - U_red).norm2() / U.norm2()

        return absolute_errors, relative_errors, speedups

    reductor = NeuralNetworkInstationaryReductor(fom, training_set, validation_set, basis_size=10,
                                                 scale_outputs=True, ann_mse=None)
    rom = reductor.reduce(hidden_layers='[30, 30, 30]', restarts=0)

    abs_errors, rel_errors, speedups = compute_errors_state(rom, reductor)

    reductor_lstm = NeuralNetworkLSTMInstationaryReductor(fom, training_set, validation_set, basis_size=10,
                                                          scale_inputs=False, scale_outputs=True, ann_mse=None)
    rom_lstm = reductor_lstm.reduce(restarts=0, number_layers=3, learning_rate=0.1)

    abs_errors_lstm, rel_errors_lstm, speedups_lstm = compute_errors_state(rom_lstm, reductor_lstm)

    def compute_errors_output(output_rom):
        outputs = []
        outputs_red = []
        outputs_speedups = []

        print(f'Performing test on set of size {len(test_set)} ...')

        for mu in test_set:
            tic = time.perf_counter()
            outputs.append(fom.compute(output=True, mu=mu)['output'][1:])
            time_fom = time.perf_counter() - tic
            tic = time.perf_counter()
            outputs_red.append(output_rom.compute(output=True, mu=mu)['output'][1:])
            time_red = time.perf_counter() - tic

            outputs_speedups.append(time_fom / time_red)

        outputs = np.squeeze(np.array(outputs))
        outputs_red = np.squeeze(np.array(outputs_red))

        outputs_absolute_errors = np.abs(outputs - outputs_red)
        outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)

        return outputs_absolute_errors, outputs_relative_errors, outputs_speedups

    output_reductor = NeuralNetworkInstationaryStatefreeOutputReductor(fom, time_steps+1, training_set,
                                                                       validation_set, validation_loss=1e-5,
                                                                       scale_outputs=True)
    output_rom = output_reductor.reduce(restarts=100)

    outputs_abs_errors, outputs_rel_errors, outputs_speedups = compute_errors_output(output_rom)

    output_reductor_lstm = NeuralNetworkLSTMInstationaryStatefreeOutputReductor(fom, time_steps+1, training_set,
                                                                                validation_set, validation_loss=None,
                                                                                scale_inputs=False, scale_outputs=True)
    output_rom_lstm = output_reductor_lstm.reduce(restarts=0, number_layers=3, hidden_dimension=50,
                                                  learning_rate=0.1)

    outputs_abs_errors_lstm, outputs_rel_errors_lstm, outputs_speedups_lstm = compute_errors_output(output_rom_lstm)

    print()
    print('Approach by Hesthaven and Ubbiali using feedforward ANNs:')
    print('=========================================================')
    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(abs_errors)}')
    print(f'Average relative error: {np.average(rel_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')

    print()
    print('Results for output approximation:')
    print(f'Average absolute error: {np.average(outputs_abs_errors)}')
    print(f'Average relative error: {np.average(outputs_rel_errors)}')
    print(f'Median of speedup: {np.median(outputs_speedups)}')

    print()
    print()
    print('Approach using long short-term memory ANNs:')
    print('===========================================')

    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(abs_errors_lstm)}')
    print(f'Average relative error: {np.average(rel_errors_lstm)}')
    print(f'Median of speedup: {np.median(speedups_lstm)}')

    print()
    print('Results for output approximation:')
    print(f'Average absolute error: {np.average(outputs_abs_errors_lstm)}')
    print(f'Average relative error: {np.average(outputs_rel_errors_lstm)}')
    print(f'Median of speedup: {np.median(outputs_speedups_lstm)}')


def create_fom(problem_number, grid_intervals, time_steps):
    print('Discretize ...')
    if problem_number == 0:
        config.require('FENICS')
        from pymor.models.examples import navier_stokes_example
        fom, plot_function = navier_stokes_example(grid_intervals, time_steps)
    elif problem_number == 1:
        problem = burgers_problem()
        f = LincombFunction(
            [ExpressionFunction('1.', 1), ConstantFunction(1., 1)],
            [ProjectionParameterFunctional('exponent'), 0.1])
        problem = problem.with_stationary_part(outputs=[('l2', f)])

        fom, _ = discretize_instationary_fv(problem, diameter=1. / grid_intervals, nt=time_steps)
        plot_function = fom.visualize
    else:
        raise ValueError(f'Unknown problem number {problem_number}')

    return fom, plot_function


if __name__ == '__main__':
    run(main)
