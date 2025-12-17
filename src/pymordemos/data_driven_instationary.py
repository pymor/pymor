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
from pymor.reductors.data_driven import DataDrivenReductor
from pymor.tools import mpi
from pymor.tools.typer import Choices


def main(
    problem_number: int = Argument(..., min=0, max=1, help='Selects the problem to solve [0 or 1].'),
    regressor: Choices('fcnn lstm vkoga gpr') = Argument(..., help="Regressor to use. Options are neural networks "
                                                                   "using PyTorch (fully-connected or long short-term "
                                                                   "memory networks), pyMOR's VKOGA algorithm or "
                                                                   "Gaussian process regression using scikit-learn."),
    grid_intervals: int = Argument(..., help='Grid interval count.'),
    time_steps: int = Argument(..., help='Number of time steps used for discretization.'),
    training_samples: int = Argument(..., help='Number of samples used for training the neural network.'),

    fv: bool = Option(False, help='Use finite volume discretization instead of finite elements.'),
    vis: bool = Option(False, help='Visualize full order solution and reduced solution for a test set.'),
    validation_ratio: float = Option(0.1, help='Ratio of training data used for validation of the neural networks.'),
    time_vectorized: bool = Option(False, help='Predict the whole time trajectory at once or iteratively.'),
    input_scaling: bool = Option(False, help='Scale the input of the regressor (i.e. the parameter).'),
    output_scaling: bool = Option(False, help='Scale the output of the regressor (i.e. reduced coefficients or output '
                                              'quantity.'),
):
    """Model order reduction with machine learning methods for instationary problems.

    Problem number 0 considers the incompressible Navier-Stokes equations in
    a two-dimensional cavity with the Reynolds number as parameter.
    The discretization is based on FEniCS.

    Problem number 1 considers a parametrized Burgers equation on a
    one-dimensional domain. The discretization is based on pyMOR's built-in
    functionality.
    """
    if (regressor == 'fcnn' or regressor == 'lstm') and not config.HAVE_TORCH:
        raise TorchMissingError
    elif (regressor == 'gpr' or input_scaling or output_scaling) and not config.HAVE_SKLEARN:
        raise SklearnMissingError

    fom, plot_function = create_fom(problem_number, grid_intervals, time_steps)

    if problem_number == 0:
        parameter_space = fom.parameters.space(1., 50.)
    else:
        parameter_space = fom.parameters.space(1., 2.)

    training_parameters = parameter_space.sample_uniformly(training_samples)
    test_parameters = parameter_space.sample_randomly(10)

    if regressor == 'fcnn':
        from pymor.algorithms.ml.nn import FullyConnectedNN, NeuralNetworkRegressor
        regressor_solution = NeuralNetworkRegressor(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                    validation_ratio=validation_ratio, tol=1e-4)
        regressor_output = NeuralNetworkRegressor(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                  validation_ratio=validation_ratio, tol=1e-4)
    elif regressor == 'lstm':
        from pymor.algorithms.ml.nn import LongShortTermMemoryNN, NeuralNetworkRegressor
        regressor_solution = NeuralNetworkRegressor(LongShortTermMemoryNN(hidden_dimension=30, number_layers=3),
                                                    validation_ratio=validation_ratio, tol=None, restarts=0)
        regressor_output = NeuralNetworkRegressor(LongShortTermMemoryNN(hidden_dimension=30, number_layers=3),
                                                  validation_ratio=validation_ratio, tol=None, restarts=0)
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
        training_outputs.extend(o for o in res['output'].T)
    training_outputs = np.array(training_outputs)

    RB, _ = pod(training_snapshots, l2_err=1e-5)
    projected_training_snapshots = training_snapshots.inner(RB)

    reductor_data_driven = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                              regressor=regressor_solution, target_quantity='solution',
                                              reduced_basis=RB, T=fom.T, time_vectorized=time_vectorized,
                                              input_scaler=input_scaler_solution, output_scaler=output_scaler_solution)
    rom_data_driven = reductor_data_driven.reduce()

    output_reductor_data_driven = DataDrivenReductor(training_parameters, training_outputs,
                                                     regressor=regressor_output, target_quantity='output',
                                                     T=fom.T, time_vectorized=time_vectorized,
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


def create_fom(problem_number, grid_intervals, time_steps):
    print('Discretize ...')
    if problem_number == 0:
        config.require('FENICS')
        fom, plot_function = discretize_navier_stokes(grid_intervals, time_steps)
    elif problem_number == 1:
        problem = burgers_problem()
        f = LincombFunction(
            [ExpressionFunction('1.', 1), ConstantFunction(1., 1)],
            [ProjectionParameterFunctional('exponent'), 0.1])
        problem = problem.with_stationary_part(outputs=[('l2', f), ('l2', ConstantFunction(1., 1))],)

        fom, _ = discretize_instationary_fv(problem, diameter=1. / grid_intervals, nt=time_steps)
        plot_function = fom.visualize
    else:
        raise ValueError(f'Unknown problem number {problem_number}')

    return fom, plot_function


def discretize_navier_stokes(n, nt):
    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(lambda: _discretize_navier_stokes(n, nt),
                             use_with=True, pickle_local_spaces=False)
        plot_function = None
    else:
        fom, plot_function = _discretize_navier_stokes(n, nt)
    return fom, plot_function


def _discretize_navier_stokes(n, nt):
    import dolfin as df
    import matplotlib.pyplot as plt

    from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
    from pymor.bindings.fenics import FenicsMatrixOperator, FenicsOperator, FenicsVectorSpace, FenicsVisualizer

    # create square mesh
    mesh = df.UnitSquareMesh(n, n)

    # create Finite Elements for the pressure and the velocity
    P = df.FiniteElement('P', mesh.ufl_cell(), 1)
    V = df.VectorElement('P', mesh.ufl_cell(), 2, dim=2)
    # create mixed element and function space
    TH = df.MixedElement([P, V])
    W = df.FunctionSpace(mesh, TH)

    # extract components of mixed space
    W_p = W.sub(0)
    W_u = W.sub(1)

    # define trial and test functions for mass matrix
    u = df.TrialFunction(W_u)
    psi_u = df.TestFunction(W_u)

    # assemble mass matrix for velocity
    mass_mat = df.assemble(df.inner(u, psi_u) * df.dx)

    # define trial and test functions
    psi_p, psi_u = df.TestFunctions(W)
    w = df.Function(W)
    p, u = df.split(w)

    # set Reynolds number, which will serve as parameter
    Re = df.Constant(1.)

    # define walls
    top_wall = 'near(x[1], 1.)'
    walls = 'near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)'

    # define no slip boundary conditions on all but the top wall
    bcu_noslip_const = df.Constant((0., 0.))
    bcu_noslip  = df.DirichletBC(W_u, bcu_noslip_const, walls)
    # define Dirichlet boundary condition for the velocity on the top wall
    bcu_lid_const = df.Constant((1., 0.))
    bcu_lid = df.DirichletBC(W_u, bcu_lid_const, top_wall)

    # fix pressure at a single point of the domain to obtain unique solutions
    pressure_point = 'near(x[0],  0.) & (x[1] <= ' + str(2./n) + ')'
    bcp_const = df.Constant(0.)
    bcp = df.DirichletBC(W_p, bcp_const, pressure_point)

    # collect boundary conditions
    bc = [bcu_noslip, bcu_lid, bcp]

    mass = -psi_p * df.div(u)
    momentum = (df.dot(psi_u, df.dot(df.grad(u), u))
                - df.div(psi_u) * p
                + 2.*(1./Re) * df.inner(df.sym(df.grad(psi_u)), df.sym(df.grad(u))))
    F = (mass + momentum) * df.dx

    df.solve(F == 0, w, bc)

    # define pyMOR operators
    space = FenicsVectorSpace(W)
    mass_op = FenicsMatrixOperator(mass_mat, W, W, name='mass')
    op = FenicsOperator(F, space, space, w, bc,
                        parameter_setter=lambda mu: Re.assign(mu['Re'].item()),
                        parameters={'Re': 1})

    # timestep size for the implicit Euler timestepper
    dt = 0.01
    ie_stepper = ImplicitEulerTimeStepper(nt=nt)

    # define initial condition and right hand side as zero
    fom_init = VectorOperator(op.range.zeros())
    rhs = VectorOperator(op.range.zeros())
    # define output functional
    output_func = VectorFunctional(op.range.ones())

    # construct instationary model
    fom = InstationaryModel(dt * nt,
                            fom_init,
                            op,
                            rhs,
                            mass=mass_op,
                            time_stepper=ie_stepper,
                            output_functional=output_func,
                            visualizer=FenicsVisualizer(space))

    def plot_fenics(w, title=''):
        v = df.Function(W)
        v.leaf_node().vector()[:] = (w.to_numpy()[:, -1]).squeeze()
        p, u  = v.split()

        fig_u = df.plot(u)
        plt.title('Velocity vector field ' + title)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(fig_u)
        plt.show()

        fig_p = df.plot(p)
        plt.title('Pressure field ' + title)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.colorbar(fig_p)
        plt.show()

    if mpi.parallel:
        return fom
    else:
        return fom, plot_fenics


if __name__ == '__main__':
    run(main)
