#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import TorchMissing


def main(
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    filepath: str = Argument(..., help='Path to the file containing the reduced basis and neural network data.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
):
    """Load reduced basis and neural network from file."""

    if not config.HAVE_TORCH:
        raise TorchMissing()

    fom = create_fom(fv, grid_intervals)

    parameter_space = fom.parameters.space((0.1, 1))

    from pymor.reductors.neural_network import NeuralNetworkReductor

    reductor = NeuralNetworkReductor(fom, [], [], l2_err=1e-5,
                                     ann_mse=1e-5)
    rom = reductor.load_model(filepath)

    test_set = parameter_space.sample_randomly(10)

    speedups = []

    import time

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

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


def create_fom(fv, grid_intervals):
    problem = StationaryProblem(
        domain=RectDomain(),
        rhs=LincombFunction(
            [ExpressionFunction('ones(x.shape[:-1]) * 10', 2, ()), ConstantFunction(1., 2)],
            [ProjectionParameterFunctional('mu'), 0.1]),
        diffusion=LincombFunction(
            [ExpressionFunction('1 - x[..., 0]', 2, ()), ExpressionFunction('x[..., 0]', 2, ())],
            [ProjectionParameterFunctional('mu'), 1]),
        dirichlet_data=LincombFunction(
            [ExpressionFunction('2 * x[..., 0]', 2, ()), ConstantFunction(1., 2)],
            [ProjectionParameterFunctional('mu'), 0.5]),
        name='2DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
    fom, _ = discretizer(problem, diameter=1. / int(grid_intervals))

    return fom


if __name__ == '__main__':
    run(main)
