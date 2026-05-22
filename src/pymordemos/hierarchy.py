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
    problem_number: Literal[0, 1],
    regressor: Literal['fcnn', 'vkoga', 'gpr'],
    grid_intervals: int,
    num_parameters: int,
    /, *,
    time_steps: int = 10,
    time_vectorized: bool = True,
    vis: bool = False,
    validation_ratio: float = 0.1,
    input_scaling: bool = False,
    output_scaling: bool = False,
):
    """Adaptive model hierarchy combining reduced basis and machine learning methods.

    Problem number 0 considers an elliptic problem and problem number 1 considers
    a parabolic problem.

    Parameters
    ----------
    problem_number
        Selects the problem to solve [0 or 1].
    regressor
        Regressor to use. Options are neural networks using PyTorch, pyMOR's VKOGA algorithm
        or Gaussian process regression using scikit-learn.
    grid_intervals
        Grid interval count.
    num_parameters
        Number of parameters to evaluate the hierarchy for.
    time_steps
        Number of time steps used for discretization (only used if `problem_number` is 1).
    time_vectorized
        Predict the whole time trajectory at once or iteratively.
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

    assert problem_number in (0, 1), f'Unknown problem number {problem_number}'

    fom, parameter_space = create_fom(problem_number, grid_intervals, time_steps)

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

    compression = None
    if problem_number == 0:
        rb_reductor = CoerciveRBReductor(fom, coercivity_estimator=ProjectionParameterFunctional('mu'))
    else:
        rb_reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product,
                                          coercivity_estimator=ProjectionParameterFunctional('diffusion'))
        compression = lambda U: pod(U, product=fom.h1_0_semi_product)[0]

    dd_reductor_parameters = {'regressor': regressor_type, 'regressor_parameters': regressor_parameters,
                              'input_scaler': input_scaler, 'output_scaler': output_scaler,
                              'time_vectorized': (problem_number == 1 and time_vectorized)}
    tol = 5e-3
    hierarchy = DDRBModelHierarchy(fom, rb_reductor, dd_reductor_parameters, tol, compression=compression)

    print(f'Performing test on parameter set of size {len(parameters)} ...')
    U = fom.solution_space.empty(reserve=len(parameters))
    timings_fom = []
    for mu in parameters:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        timings_fom.append(time.perf_counter() - tic)

    U_red = fom.solution_space.empty(reserve=len(parameters))
    timings_red = []
    used_models = []
    estimated_errors = []
    for mu in parameters:
        tic = time.perf_counter()
        data = hierarchy.compute(solution=True, solution_error_estimate=True, mu=mu)
        timings_red.append(time.perf_counter() - tic)
        U_red.append(data['solution'])
        used_models.append(data['used_model'])
        estimated_errors.append(np.max(data['estimated_error']))

    timings_fom = np.array(timings_fom)
    timings_red = np.array(timings_red)
    estimated_errors = np.array(estimated_errors)

    print(f'Mean speedup: {np.mean(timings_fom / timings_red)}')
    norms = U.norm()
    relative_errors = (U - U_red).norm() / np.where(norms > 0, norms, 1.)
    print(f'Mean errors: {np.mean(relative_errors)}')

    n = len(used_models)
    for model in ('FOM', 'RB', 'DD'):
        count = used_models.count(model)
        print(f'\t{model}: {count}\t(ratio: {count/n*100:.2f}%)')

    if vis:
        model_colors = {'FOM': 'C0', 'RB': 'C1', 'DD': 'C2'}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Model usage counts
        models = ('FOM', 'RB', 'DD')
        counts = [used_models.count(m) for m in models]
        axes[0].bar(models, counts, color=[model_colors[m] for m in models])
        axes[0].set_ylabel('count')
        axes[0].set_title('Model usage')

        # Runtime box plots
        timing_data = [timings_fom]
        labels = ['FOM (direct)']
        colors = [model_colors['FOM']]
        for model in models:
            idx = [i for i, m in enumerate(used_models) if m == model]
            if idx:
                timing_data.append(timings_red[idx])
                labels.append(f'Hierarchy ({model})')
                colors.append(model_colors[model])
        bplot = axes[1].boxplot(timing_data, labels=labels, patch_artist=True)
        for patch, color in zip(bplot['boxes'], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_yscale('log')
        axes[1].set_ylabel('time [s]')
        axes[1].set_title('Runtimes')

        # Estimated errors
        for model, marker in (('DD', '*'), ('RB', '.')):
            idx = [i for i, m in enumerate(used_models) if m == model]
            if idx:
                axes[2].plot(idx, estimated_errors[idx], marker,
                             color=model_colors[model], label=model)
        axes[2].set_xlabel('parameter index')
        axes[2].set_ylabel('error estimate')
        axes[2].semilogy()
        axes[2].legend()
        axes[2].set_title('Estimated errors')

        fig.tight_layout()
        plt.show()


def create_fom(problem_number, grid_intervals, time_steps):
    print('Discretize ...')
    if problem_number == 0:
        from pymor.models.examples import two_dimensional_parametric_diffusion
        fom = two_dimensional_parametric_diffusion(grid_intervals=grid_intervals)
        parameter_space = fom.parameters.space((0.1, 1))
    else:
        stationary_part = text_problem(text='p')
        problem = InstationaryProblem(stationary_part, initial_data=ConstantFunction(0., 2), T=1.)
        fom, _ = discretize_instationary_cg(problem, diameter=5., nt=time_steps)
        parameter_space = fom.parameters.space((0.1, 1))

    return fom, parameter_space


if __name__ == '__main__':
    app()
