# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import matplotlib.pyplot as plt
import numpy as np
from typer import Argument, Option, run

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGARegressor
from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import SklearnMissingError, TorchMissingError
from pymor.reductors.data_driven import DataDrivenReductor
from pymor.tools.typer import Choices


def main(
    regressor: Choices('fcnn vkoga gpr') = Argument(..., help="Regressor to use. Options are neural networks "
                                                              "using PyTorch, pyMOR's VKOGA algorithm or Gaussian "
                                                              "process regression using scikit-learn."),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
    validation_ratio: float = Option(0.1, help='Ratio of training data used for validation of the neural networks.'),
    input_scaling: bool = Option(False, help='Scale the input of the regressor (i.e. the parameter).'),
    output_scaling: bool = Option(False, help='Scale the output of the regressor (i.e. reduced coefficients or output '
                                              'quantity.'),
):
    """Model order reduction with machine learning methods (approach by Hesthaven and Ubbiali)."""
    if regressor == 'fcnn' and not config.HAVE_TORCH:
        raise TorchMissingError
    elif (regressor == 'gpr' or input_scaling or output_scaling) and not config.HAVE_SKLEARN:
        raise SklearnMissingError

    fom = create_fom(fv, grid_intervals)

    parameter_space = fom.parameters.space((0.1, 1))

    training_parameters = parameter_space.sample_randomly(training_samples)
    test_parameters = parameter_space.sample_randomly(10)

    if regressor == 'fcnn':
        from pymor.algorithms.ml.nn import FullyConnectedNN, NeuralNetworkRegressor
        regressor_solution = NeuralNetworkRegressor(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                    validation_ratio=validation_ratio, tol=1e-5)
        regressor_output = NeuralNetworkRegressor(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                  validation_ratio=validation_ratio, tol=1e-5)
    elif regressor == 'vkoga':
        kernel = GaussianKernel(length_scale=1.0)
        regressor_solution = VKOGARegressor(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
        regressor_output = VKOGARegressor(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
    elif regressor == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        regressor_solution = GaussianProcessRegressor()
        regressor_output = GaussianProcessRegressor()

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

    reductor = DataDrivenReductor(training_parameters[:1], projected_training_snapshots[:1],
                                              regressor=regressor_solution, target_quantity='solution',
                                              reduced_basis=RB,
                                              input_scaler=input_scaler_solution, output_scaler=output_scaler_solution)
    rom = reductor.reduce()

    output_reductor = DataDrivenReductor(training_parameters[:1], training_outputs[:1],
                                                     regressor=regressor_output, target_quantity='output',
                                                     input_scaler=input_scaler_output,
                                                     output_scaler=output_scaler_output)
    output_rom = output_reductor.reduce()

    print(f'Performing test on parameter set of size {len(test_parameters)} ...')
    U = fom.solution_space.empty(reserve=len(test_parameters))
    timings_fom = []
    outputs = []
    output_timings_fom = []
    for mu in test_parameters:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        time_fom = time.perf_counter() - tic
        timings_fom.append(time_fom)

        tic = time.perf_counter()
        outputs.append(fom.output(mu=mu))
        output_time_fom = time.perf_counter() - tic
        output_timings_fom.append(output_time_fom)

    error_statistics = []
    output_error_statistics = []

    speedups = []
    outputs_speedups = []
    for mu, ts, to in zip(training_parameters[1:], projected_training_snapshots[1:], training_outputs[1:],
                          strict=False):
        print(f'Extending training data by {mu}...')
        reductor.extend_training_data([mu], ts.reshape((1, -1)))
        reductor.reduce()
        output_reductor.extend_training_data([mu], to.reshape((1, -1)))
        output_reductor.reduce()

        U_red = fom.solution_space.empty(reserve=len(test_parameters))
        timings_red = []

        outputs_red = []
        output_timings_red = []

        for mu in test_parameters:
            tic = time.perf_counter()
            U_red.append(reductor.reconstruct(rom.solve(mu)))
            time_red = time.perf_counter() - tic
            timings_red.append(time_red)

            tic = time.perf_counter()
            outputs_red.append(output_rom.output(mu=mu))
            output_time_red = time.perf_counter() - tic
            output_timings_red.append(output_time_red)

        speedups.append(np.array(timings_fom) / np.array(timings_red))
        outputs_speedups.append(np.array(output_timings_fom) / np.array(output_timings_red))

        outputs = np.squeeze(np.array(outputs))
        outputs_red = np.squeeze(np.array(outputs_red))

        outputs_relative_errors = np.abs(outputs - outputs_red) / np.abs(outputs)
        output_error_statistics.append(outputs_relative_errors)
        relative_errors = (U - U_red).norm() / U.norm()
        error_statistics.append(relative_errors)

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_xlabel('number of training data points')
    axs[0, 0].set_ylabel('error over test set')
    axs[0, 0].boxplot(np.array(error_statistics).T, patch_artist=True)
    axs[0, 0].semilogy()
    axs[0, 0].set_title('Errors in state approximation')
    axs[0, 1].set_xlabel('number of training data points')
    axs[0, 1].set_ylabel('speedups')
    axs[0, 1].boxplot(np.array(speedups).T, patch_artist=True)
    axs[0, 1].set_title('Speedups of the state approximation')
    axs[1, 0].set_xlabel('number of training data points')
    axs[1, 0].set_ylabel('error over test set')
    output_error_statistics = np.linalg.norm(np.array(output_error_statistics), axis=-1)
    axs[1, 0].boxplot(output_error_statistics.T, patch_artist=True)
    axs[1, 0].semilogy()
    axs[1, 0].set_title('Errors in output approximation')
    axs[1, 1].set_xlabel('number of training data points')
    axs[1, 1].set_ylabel('speedups')
    axs[1, 1].boxplot(np.array(outputs_speedups).T, patch_artist=True)
    axs[1, 1].set_title('Speedups of the output approximation')
    plt.show()

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
