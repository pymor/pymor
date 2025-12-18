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
from pymor.solvers.newton import NewtonSolver
from pymor.tools.typer import Choices

DIM = 2
GRID_INTERVALS = 50
FENICS_ORDER = 1


def main(
    regressor: Choices('fcnn vkoga gpr') = Argument(..., help="Regressor to use. Options are neural networks "
                                                              "using PyTorch, pyMOR's VKOGA algorithm or Gaussian "
                                                              "process regression using scikit-learn."),
    training_samples: int = Argument(..., help='Number of samples used for computing the reduced basis and '
                                               'training the regressor.'),

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

    fom, parameter_space = discretize_fenics()

    training_parameters = parameter_space.sample_uniformly(training_samples)
    test_parameters = parameter_space.sample_randomly(1)

    if regressor == 'fcnn':
        from pymor.algorithms.ml.nn import FullyConnectedNN, NeuralNetworkRegressor
        regressor_solution = NeuralNetworkRegressor(FullyConnectedNN(hidden_layers=[30, 30, 30]),
                                                    validation_ratio=validation_ratio, tol=1e-5)
    elif regressor == 'vkoga':
        kernel = GaussianKernel(length_scale=1.0)
        regressor_solution = VKOGARegressor(kernel=kernel, criterion='fp', max_centers=30, tol=1e-6, reg=1e-12)
    elif regressor == 'gpr':
        from sklearn.gaussian_process import GaussianProcessRegressor
        regressor_solution = GaussianProcessRegressor()

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

    training_snapshots = fom.solution_space.empty(reserve=len(training_parameters))
    for mu in training_parameters:
        training_snapshots.append(fom.solve(mu))

    RB, _ = pod(training_snapshots, l2_err=1e-5)
    projected_training_snapshots = training_snapshots.inner(RB)

    reductor = DataDrivenReductor(training_parameters, projected_training_snapshots,
                                  regressor=regressor_solution, target_quantity='solution',
                                  reduced_basis=RB, input_scaler=input_scaler, output_scaler=output_scaler)
    rom = reductor.reduce()

    speedups = []

    print(f'Performing test on parameters of size {len(test_parameters)} ...')

    U = fom.solution_space.empty(reserve=len(test_parameters))
    U_red = fom.solution_space.empty(reserve=len(test_parameters))

    for mu in test_parameters:
        tic = time.perf_counter()
        U.append(fom.solve(mu))
        time_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        U_red.append(reductor.reconstruct(rom.solve(mu)))
        time_red = time.perf_counter() - tic

        speedups.append(time_fom / time_red)

    absolute_errors = (U - U_red).norm()
    relative_errors = (U - U_red).norm() / U.norm()

    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')


def discretize_fenics():
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(_discretize_fenics, use_with=True, pickle_local_spaces=False)
    else:
        fom = _discretize_fenics()
    return fom, fom.parameters.space((0, 1000.))


def _discretize_fenics():
    import dolfin as df

    if DIM == 2:
        mesh = df.UnitSquareMesh(GRID_INTERVALS, GRID_INTERVALS)
    elif DIM == 3:
        mesh = df.UnitCubeMesh(GRID_INTERVALS, GRID_INTERVALS, GRID_INTERVALS)
    else:
        raise NotImplementedError

    V = df.FunctionSpace(mesh, 'CG', FENICS_ORDER)

    g = df.Constant(1.0)
    c = df.Constant(1.)

    class DirichletBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 1.0) < df.DOLFIN_EPS and on_boundary
    db = DirichletBoundary()
    bc = df.DirichletBC(V, g, db)

    u = df.Function(V)
    v = df.TestFunction(V)
    f = df.Expression('x[0]*sin(x[1])', degree=2)
    F = df.inner((1 + c*u**2)*df.grad(u), df.grad(v))*df.dx - f*v*df.dx

    df.solve(F == 0, u, bc,
             solver_parameters={'newton_solver': {'relative_tolerance': 1e-6}})

    from pymor.bindings.fenics import FenicsOperator, FenicsVectorSpace

    space = FenicsVectorSpace(V)
    op = FenicsOperator(F, space, space, u, (bc,),
                        parameter_setter=lambda mu: c.assign(mu['c'].item()),
                        parameters={'c': 1},
                        solver=NewtonSolver(rtol=1e-6))
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs)

    return fom


if __name__ == '__main__':
    run(main)
