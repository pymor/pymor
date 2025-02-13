#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from typer import Argument, Option, run

from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.basic import *
from pymor.tools.typer import Choices

# parameters for high-dimensional models
GRID_INTERVALS = 50
FENICS_ORDER = 2
NT = 50
DT = 1. / NT


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main(
    backend: Choices('pymor fenics') = Argument(..., help='Discretization toolkit to use.'),
    alg: Choices('greedy adaptive_greedy pod') = Argument(..., help='The model reduction algorithm to use.'),
    snapshots: int = Argument(
        ...,
        help='greedy/pod: number of training set parameters\n\n'
             'adaptive_greedy: size of validation set.'
    ),
    rbsize: int = Argument(..., help='Size of the reduced basis.'),
    test: int = Argument(..., help='Number of test parameters for reduction error estimation.'),
    pickle: bool = Option(
        True,
        help='Pickle reduced model and error analysis results to files.'
    ),
    plot_err: bool = Option(False, help='Plot error'),
    plot_error_sequence: bool = Option(False, help='Plot reduction error vs. basis size.'),
):
    """Reduced basis approximation of the heat equation."""
    # discretize
    ############
    if backend == 'pymor':
        fom = discretize_pymor()
    elif backend == 'fenics':
        fom = discretize_fenics()
    else:
        raise NotImplementedError
    parameter_space = fom.parameters.space(1, 100)

    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
    reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)

    # generate reduced model
    ########################
    if alg == 'greedy':
        rom = reduce_greedy(fom, reductor, parameter_space, snapshots, rbsize)
    elif alg == 'adaptive_greedy':
        rom = reduce_adaptive_greedy(fom, reductor, parameter_space, snapshots, rbsize)
    elif alg == 'pod':
        rom = reduce_pod(fom, reductor, parameter_space, snapshots, rbsize)
    else:
        raise NotImplementedError

    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(
        rom, fom=fom, reductor=reductor, error_estimator=True,
        error_norms=[lambda U: DT * np.sqrt(np.sum(fom.h1_0_semi_norm(U)[1:]**2))],
        error_norm_names=['l^2-h^1'],
        condition=False, test_mus=parameter_space.sample_randomly(test)
    )

    # show results
    ##############
    print(results['summary'])
    if plot_error_sequence:
        plot_reduction_error_analysis(results)

    # write results to disk
    #######################
    if pickle:
        from pymor.core.pickle import dump
        with open('reduced_model.out', 'wb') as f:
            dump(rom, f)
        with open('results.out', 'wb') as f:
            dump(results, f)

    # visualize reduction error for worst-approximated mu
    #####################################################
    if plot_err:
        mumax = results['max_error_mus'][0, -1]
        U = fom.solve(mumax)
        U_RB = reductor.reconstruct(rom.solve(mumax))
        if backend == 'fenics':  # right now the fenics visualizer does not support time trajectories
            U = U[len(U) - 1].copy()
            U_RB = U_RB[len(U_RB) - 1].copy()
        fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                      separate_colorbars=True)

    return results


####################################################################################################
# High-dimensional models                                                                          #
####################################################################################################


def discretize_pymor():
    from pymor.models.examples import heat_equation_example
    fom = heat_equation_example(grid_intervals=GRID_INTERVALS, nt=NT)
    fom.enable_caching('disk')
    return fom


def discretize_fenics():
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        return mpi_wrap_model(_discretize_fenics, use_with=True, pickle_local_spaces=False)
    else:
        return _discretize_fenics()


def _discretize_fenics():

    # assemble system matrices - FEniCS code
    ########################################

    import dolfin as df

    # discrete function space
    mesh = df.UnitSquareMesh(GRID_INTERVALS, GRID_INTERVALS, 'crossed')
    V = df.FunctionSpace(mesh, 'Lagrange', FENICS_ORDER)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    # data functions
    bottom_diffusion = df.Expression('(x[0] > 0.45) * (x[0] < 0.55) * (x[1] < 0.7) * 1.',
                                     element=df.FunctionSpace(mesh, 'DG', 0).ufl_element())
    top_diffusion = df.Expression('(x[0] > 0.35) * (x[0] < 0.40) * (x[1] > 0.3) * 1. +'
                                  '(x[0] > 0.60) * (x[0] < 0.65) * (x[1] > 0.3) * 1.',
                                  element=df.FunctionSpace(mesh, 'DG', 0).ufl_element())
    initial_data = df.Expression('(x[0] > 0.45) * (x[0] < 0.55) * (x[1] < 0.7) * 10.',
                                 element=df.FunctionSpace(mesh, 'DG', 0).ufl_element())
    neumann_data = df.Expression('(x[0] > 0.45) * (x[0] < 0.55) * 1000.',
                                 element=df.FunctionSpace(mesh, 'DG', 0).ufl_element())

    # assemble matrices and vectors
    l2_mat = df.assemble(df.inner(u, v) * df.dx)
    l2_0_mat = l2_mat.copy()
    mass_mat = l2_0_mat.copy()
    h1_mat = df.assemble(df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)
    h1_0_mat = h1_mat.copy()
    bottom_mat = df.assemble(bottom_diffusion * df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)
    top_mat = df.assemble(top_diffusion * df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)
    u0 = df.project(initial_data, V).vector()
    f = df.assemble(neumann_data * v * df.ds)

    # boundary treatment
    def dirichlet_boundary(x, on_boundary):
        tol = 1e-14
        return on_boundary and (abs(x[0]) < tol or abs(x[0] - 1) < tol or abs(x[1] - 1) < tol)

    bc = df.DirichletBC(V, df.Constant(0.), dirichlet_boundary)
    bc.apply(l2_0_mat)
    bc.zero(mass_mat)
    bc.apply(h1_0_mat)
    bc.zero(bottom_mat)
    bc.zero(top_mat)
    bc.apply(f)
    bc.apply(u0)

    # wrap everything as a pyMOR model
    ##################################

    from pymor.bindings.fenics import FenicsMatrixOperator, FenicsVectorSpace, FenicsVisualizer

    fom = InstationaryModel(
        T=1.,

        initial_data=FenicsVectorSpace(V).make_array([u0]),

        operator=LincombOperator([FenicsMatrixOperator(h1_0_mat, V, V),
                                  FenicsMatrixOperator(bottom_mat, V, V),
                                  FenicsMatrixOperator(top_mat, V, V)],
                                 [1.,
                                  100. - 1.,
                                  ExpressionParameterFunctional('top[0] - 1.', {'top': 1})]),

        rhs=VectorOperator(FenicsVectorSpace(V).make_array([f])),

        mass=FenicsMatrixOperator(mass_mat, V, V, name='mass'),

        products={'l2': FenicsMatrixOperator(l2_mat, V, V, name='l2'),
                  'l2_0': FenicsMatrixOperator(l2_0_mat, V, V, name='l2_0'),
                  'h1': FenicsMatrixOperator(h1_mat, V, V, name='h1'),
                  'h1_0_semi': FenicsMatrixOperator(h1_0_mat, V, V, name='h1_0_semi')},

        time_stepper=ImplicitEulerTimeStepper(nt=NT),

        visualizer=FenicsVisualizer(FenicsVectorSpace(V))
    )

    return fom


####################################################################################################
# Reduction algorithms                                                                             #
####################################################################################################


def reduce_greedy(fom, reductor, parameter_space, snapshots, basis_size):

    training_set = parameter_space.sample_uniformly(snapshots)
    pool = new_parallel_pool()

    greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=basis_size, pool=pool)

    return greedy_data['rom']


def reduce_adaptive_greedy(fom, reductor, parameter_space, validation_mus, basis_size):

    pool = new_parallel_pool()

    greedy_data = rb_adaptive_greedy(fom, reductor, parameter_space,
                                     validation_mus=validation_mus,
                                     max_extensions=basis_size, pool=pool)

    return greedy_data['rom']


def reduce_pod(fom, reductor, parameter_space, snapshots, basis_size):

    training_set = parameter_space.sample_uniformly(snapshots)

    snapshots = fom.operator.source.empty()
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    basis, singular_values = pod(snapshots, modes=basis_size, product=fom.h1_0_semi_product)
    reductor.extend_basis(basis, method='trivial')

    rom = reductor.reduce()

    return rom


if __name__ == '__main__':
    run(main)
