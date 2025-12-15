# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
from typer import Argument, Option, run

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGAEstimator
from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import SklearnMissingError, TorchMissingError
from pymor.reductors.data_driven import DataDrivenReductor
from pymor.tools.typer import Choices


def main(
    estimator: Choices('fcnn vkoga gpr') = Argument(..., help="Estimator to use. Options are neural networks "
                                                              "using PyTorch, pyMOR's VKOGA algorithm or Gaussian "
                                                              "process regression using scikit-learn."),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
    validation_ratio: float = Option(0.1, help='Ratio of training data used for validation of the neural networks.'),
    input_scaling: bool = Option(False, help='Scale the input of the estimator (i.e. the parameter).'),
    output_scaling: bool = Option(False, help='Scale the output of the estimator (i.e. reduced coefficients or output '
                                              'quantity.'),
):
    """Model order reduction with machine learning methods (approach by Hesthaven and Ubbiali)."""
    if estimator == 'fcnn' and not config.HAVE_TORCH:
        raise TorchMissingError
    elif (estimator == 'gpr' or input_scaling or output_scaling) and not config.HAVE_SKLEARN:
        raise SklearnMissingError

    fom = create_fom(fv, grid_intervals)

    parameter_space = fom.parameters.space((0.1, 1))

    training_parameters = parameter_space.sample_uniformly(training_samples)
    test_parameters = parameter_space.sample_randomly(10)

    if estimator == 'fcnn':
        from pymor.algorithms.ml.nn import FullyConnectedNN, NeuralNetworkEstimator
        estimator_solution = NeuralNetworkEstimator(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                    validation_ratio=validation_ratio, tol=1e-5)
        estimator_output = NeuralNetworkEstimator(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                  validation_ratio=validation_ratio, tol=1e-5)
    elif estimator == 'vkoga':
        kernel = GaussianKernel(length_scale=1.0)
        estimator_solution = VKOGAEstimator(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
        estimator_output = VKOGAEstimator(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
    elif estimator == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        estimator_solution = GaussianProcessRegressor()
        estimator_output = GaussianProcessRegressor()

    if input_scaling or output_scaling:
        from sklearn.preprocessing import MinMaxScaler
    if input_scaling:
        input_scaler_solution = MinMaxScaler()
        input_scaler_output = MinMaxScaler()
    else:
        input_scaler_solution = None
        input_scaler_output = None
    if output_scaling:
        output_scaler_solution = MinMaxScaler()
        output_scaler_output = MinMaxScaler()
    else:
        output_scaler_solution = None
        output_scaler_output = None

    training_outputs = []
    training_snapshots = fom.solution_space.empty(reserve=len(training_parameters))
    for mu in training_parameters:
        res = fom.compute(solution=True, output=True, mu=mu)
        training_snapshots.append(res['solution'])
        training_outputs.append(res['output'][:, 0])
    training_outputs = np.array(training_outputs)

    RB, _ = pod(training_snapshots, l2_err=1e-5)
    projected_training_snapshots = training_snapshots.inner(RB)

    reductor_data_driven = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                              estimator=estimator_solution, target_quantity='solution',
                                              reduced_basis=RB,
                                              input_scaler=input_scaler_solution, output_scaler=output_scaler_solution)
    rom_data_driven = reductor_data_driven.reduce()

    output_reductor_data_driven = DataDrivenReductor(training_parameters, training_outputs,
                                                     estimator=estimator_output, target_quantity='output',
                                                     input_scaler=input_scaler_output,
                                                     output_scaler=output_scaler_output)
    output_rom_data_driven = output_reductor_data_driven.reduce()


    print(f'Performing test on parameter set of size {len(test_parameters)} ...')

    U = fom.solution_space.empty(reserve=len(test_parameters))
    U_red_data_driven = fom.solution_space.empty(reserve=len(test_parameters))
    speedups_data_driven = []

    outputs = []
    outputs_red = []
    outputs_speedups = []

    for mu in test_parameters:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        U_red_data_driven.append(reductor_data_driven.reconstruct(rom_data_driven.solve(mu)))
        time_red_data_driven = time.perf_counter() - tic

        speedups_data_driven.append(time_fom / time_red_data_driven)

        tic = time.perf_counter()
        outputs.append(fom.output(mu=mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        outputs_red.append(output_rom_data_driven.output(mu=mu))
        time_red = time.perf_counter() - tic

        outputs_speedups.append(time_fom / time_red)

    outputs = np.squeeze(np.array(outputs))
    outputs_red = np.squeeze(np.array(outputs_red))

    outputs_absolute_errors = np.abs(outputs - outputs_red)
    outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)

    absolute_errors_data_driven = (U - U_red_data_driven).norm()
    relative_errors_data_driven = (U - U_red_data_driven).norm() / U.norm()

    if vis:
        fom.visualize((U, U_red_data_driven),
                      legend=('Full solution', 'Reduced solution (data-driven)'))

    print()
    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(absolute_errors_data_driven)}')
    print(f'Average relative error: {np.average(relative_errors_data_driven)}')
    print(f'Median of speedup: {np.median(speedups_data_driven)}')

    print()
    print('Results for output approximation:')
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
