# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time

import numpy as np
from typer import Argument, Option, run

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGARegressor
from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import SklearnMissingError, TorchMissingError
from pymor.models.hierarchy import DDRBModelHierarchy
from pymor.tools.typer import Choices


def main(
    regressor: Choices('fcnn vkoga gpr') = Argument(..., help="Regressor to use. Options are neural networks "
                                                              "using PyTorch, pyMOR's VKOGA algorithm or Gaussian "
                                                              "process regression using scikit-learn."),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    num_parameters: int = Argument(..., help='Number of parameters to evaluate the hierarchy for.'),

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

    parameters = parameter_space.sample_randomly(num_parameters)

    if regressor == 'fcnn':
        from pymor.algorithms.ml.nn import FullyConnectedNN, NeuralNetworkRegressor
        regressor_type = NeuralNetworkRegressor
        regressor_parameters = {'network': FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                'validation_ratio': validation_ratio}
    elif regressor == 'vkoga':
        kernel = GaussianKernel(length_scale=1.0)
        regressor_type = VKOGARegressor
        regressor_parameters = {'kernel': kernel, 'criterion': 'fp', 'max_centers': 30, 'tol': 1e-6, 'reg': 1e-12}
    elif regressor == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        regressor_type = GaussianProcessRegressor
        regressor_parameters = {}

    if input_scaling or output_scaling:
        from sklearn.preprocessing import MinMaxScaler
    if input_scaling:
        input_scaler = MinMaxScaler()
    else:
        input_scaler = None
    if output_scaling:
        output_scaler = MinMaxScaler()
    else:
        output_scaler = None

    rb_reductor = CoerciveRBReductor(fom, coercivity_estimator=ProjectionParameterFunctional('mu'))
    dd_reductor_parameters = {'regressor': regressor_type, 'regressor_parameters': regressor_parameters,
                              'input_scaler': input_scaler, 'output_scaler': output_scaler}
    tol = 5e-3
    hierarchy = DDRBModelHierarchy(fom, rb_reductor, dd_reductor_parameters, tol)

    print(f'Performing test on parameter set of size {len(parameters)} ...')
    U = fom.solution_space.empty(reserve=len(parameters))
    timings_fom = []
    for mu in parameters:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        time_fom = time.perf_counter() - tic
        timings_fom.append(time_fom)

    timings_red = []
    U_red = fom.solution_space.empty(reserve=len(parameters))
    for mu in parameters:
        tic = time.perf_counter()
        U_red.append(hierarchy.solve(mu))
        time_red = time.perf_counter() - tic
        timings_red.append(time_red)

    print(f'Mean speedup: {np.mean(np.array(timings_fom) / np.array(timings_red))}')

    relative_errors = (U - U_red).norm() / U.norm()
    print(f'Mean errors: {np.mean(relative_errors)}')

    hierarchy.print_summary()
    hierarchy.plot_summary()


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
        name='2DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
    fom, _ = discretizer(problem, diameter=1. / int(grid_intervals))

    return fom


if __name__ == '__main__':
    run(main)
