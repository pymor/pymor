#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import time
import numpy as np
from typer import Argument, Option, run

from pymor.basic import *
from pymor.core.config import config
from pymor.reductors.neural_network import (NeuralNetworkInstationaryReductor,
                                            NeuralNetworkInstationaryStatefreeOutputReductor)
from pymor.tools import mpi


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
    config.require("TORCH")

    fom, plot_function = create_fom(problem_number, grid_intervals, time_steps)

    if problem_number == 0:
        parameter_space = fom.parameters.space(1., 50.)
    else:
        parameter_space = fom.parameters.space(1., 2.)

    training_set = parameter_space.sample_uniformly(training_samples)
    validation_set = parameter_space.sample_randomly(validation_samples)

    reductor = NeuralNetworkInstationaryReductor(fom, training_set, validation_set, basis_size=10,
                                                 scale_outputs=True, ann_mse=None)
    rom = reductor.reduce(hidden_layers='[30, 30, 30]', restarts=0)

    test_set = parameter_space.sample_randomly(10)

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

    output_reductor = NeuralNetworkInstationaryStatefreeOutputReductor(fom, time_steps+1, training_set,
                                                                       validation_set, validation_loss=1e-5,
                                                                       scale_outputs=True)
    output_rom = output_reductor.reduce(restarts=100)

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

    print('Results for state approximation:')
    print(f'Average absolute error: {np.average(absolute_errors)}')
    print(f'Average relative error: {np.average(relative_errors)}')
    print(f'Median of speedup: {np.median(speedups)}')

    print()
    print('Results for output approximation:')
    print(f'Average absolute error: {np.average(outputs_absolute_errors)}')
    print(f'Average relative error: {np.average(outputs_relative_errors)}')
    print(f'Median of speedup: {np.median(outputs_speedups)}')


def create_fom(problem_number, grid_intervals, time_steps):
    print('Discretize ...')
    if problem_number == 0:
        config.require("FENICS")
        fom, plot_function = discretize_navier_stokes(grid_intervals, time_steps)
    elif problem_number == 1:
        problem = burgers_problem()
        f = LincombFunction(
            [ExpressionFunction('1.', 1), ConstantFunction(1., 1)],
            [ProjectionParameterFunctional('exponent'), 0.1])
        problem = problem.with_stationary_part(outputs=[('l2', f)])

        fom, _ = discretize_instationary_fv(problem, diameter=1. / grid_intervals, nt=time_steps)
        plot_function = fom.visualize
    else:
        assert False

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
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsOperator, FenicsVisualizer, FenicsMatrixOperator
    from pymor.algorithms.timestepping import ImplicitEulerTimeStepper

    import dolfin as df
    import matplotlib.pyplot as plt

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
    top_wall = "near(x[1], 1.)"
    walls = "near(x[0], 0.) | near(x[0], 1.) | near(x[1], 0.)"

    # define no slip boundary conditions on all but the top wall
    bcu_noslip_const = df.Constant((0., 0.))
    bcu_noslip  = df.DirichletBC(W_u, bcu_noslip_const, walls)
    # define Dirichlet boundary condition for the velocity on the top wall
    bcu_lid_const = df.Constant((1., 0.))
    bcu_lid = df.DirichletBC(W_u, bcu_lid_const, top_wall)

    # fix pressure at a single point of the domain to obtain unique solutions
    pressure_point = "near(x[0],  0.) & (x[1] <= " + str(2./n) + ")"
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
        v.leaf_node().vector()[:] = (w.to_numpy()[-1, :]).squeeze()
        p, u  = v.split()

        fig_u = df.plot(u)
        plt.title("Velocity vector field " + title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.colorbar(fig_u)
        plt.show()

        fig_p = df.plot(p)
        plt.title("Pressure field " + title)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.colorbar(fig_p)
        plt.show()

    if mpi.parallel:
        return fom
    else:
        return fom, plot_fenics


if __name__ == '__main__':
    run(main)
