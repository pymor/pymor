# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from cyclopts import App

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGARegressor
from pymor.basic import *
from pymor.core.config import config
from pymor.core.exceptions import SklearnMissingError, TorchMissingError
from pymor.models.hierarchy import DDRBModelHierarchy

app = App(help_on_error=True)

@app.default
def main(
    regressor: Literal['fcnn', 'vkoga', 'gpr'],
    grid_intervals: int,
    num_parameters: int,
    /, *,
    fv: bool = False,
    vis: bool = False,
    validation_ratio: float = 0.1,
    input_scaling: bool = False,
    output_scaling: bool = False,
):
    """Model order reduction with machine learning methods (approach by Hesthaven and Ubbiali).

    Parameters
    ----------
    regressor
        Regressor to use. Options are neural networks using PyTorch, pyMOR's VKOGA algorithm
        or Gaussian process regression using scikit-learn.
    grid_intervals
        Grid interval count.
    num_parameters
        Number of parameters to evaluate the hierarchy for.
    fv
        Use finite volume discretization instead of finite elements.
    vis
        Visualize estimated errors for the queried parameters.
    validation_ratio
        Ratio of training data used for validation of the neural networks.
    input_scaling
        Scale the input of the regressor (i.e. the parameter).
    output_scaling
        Scale the output of the regressor (i.e. reduced coefficients or output quantity).
    """
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
    used_models = []
    estimated_errors = []
    for mu in parameters:
        tic = time.perf_counter()
        data = hierarchy.compute(solution=True, solution_error_estimate=True, mu=mu)
        time_red = time.perf_counter() - tic
        U_red.append(data['solution'])
        timings_red.append(time_red)
        used_models.append(data['_used_model'])
        estimated_errors.append(data['_estimated_error'][0])

    print(f'Mean speedup: {np.mean(np.array(timings_fom) / np.array(timings_red))}')

    relative_errors = (U - U_red).norm() / U.norm()
    print(f'Mean errors: {np.mean(relative_errors)}')

    n = len(used_models)
    for model in ('FOM', 'RB', 'ML'):
        count = used_models.count(model)
        print(f'\t{model}: {count}\t(ratio: {count/n*100:.2f}%)')

    if vis:
        estimated_errors = np.array(estimated_errors)
        fig, ax = plt.subplots()
        for model, marker in (('ML', '*'), ('RB', '.')):
            idx = [i for i, m in enumerate(used_models) if m == model]
            if idx:
                ax.plot(idx, estimated_errors[idx], marker, label=model)
        ax.set_xlabel('parameter index')
        ax.set_ylabel('error estimate')
        ax.semilogy()
        ax.legend()
        ax.set_title('Estimated errors')
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
        name='2DProblem'
    )

    print('Discretize ...')
    discretizer = discretize_stationary_fv if fv else discretize_stationary_cg
    fom, _ = discretizer(problem, diameter=1. / int(grid_intervals))

    return fom


if __name__ == '__main__':
    app()
