#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Reduced basis approximation of the heat equation.

Usage:
  parabolic_mor.py BACKEND ALG SNAPSHOTS RBSIZE TEST

Arguments:
  BACKEND    Discretization toolkit to use (pymor, fenics).

  ALG        The model reduction algorithm to use
             (greedy, adaptive_greedy, pod).

  SNAPSHOTS  greedy/pod:      number of training set parameters
             adaptive_greedy: size of validation set.

  RBSIZE     Size of the reduced basis.
  TEST       Number of test parameters for reduction error estimation.
"""

from functools import partial    # fix parameters of given function

import numpy as np

from pymor.basic import *        # most common pyMOR functions and classes
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper


# parameters for high-dimensional models
GRID_INTERVALS = 100
FENICS_ORDER = 2
NT = 100
DT = 1. / NT


####################################################################################################
# High-dimensional models                                                                          #
####################################################################################################


def discretize_pymor():

    # setup analytical problem
    problem = InstationaryProblem(

        StationaryProblem(
            domain=RectDomain(top='dirichlet', bottom='neumann'),

            diffusion=LincombFunction(
                [ConstantFunction(1., dim_domain=2),
                 ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * (x[..., 1] < 0.7) * 1.',
                                    dim_domain=2),
                 ExpressionFunction('(x[..., 0] > 0.35) * (x[..., 0] < 0.40) * (x[..., 1] > 0.3) * 1. + '
                                    '(x[..., 0] > 0.60) * (x[..., 0] < 0.65) * (x[..., 1] > 0.3) * 1.',
                                    dim_domain=2)],
                [1.,
                 100. - 1.,
                 ExpressionParameterFunctional('top[0] - 1.', {'top': 1})]
            ),

            rhs=ConstantFunction(value=100., dim_domain=2) * ExpressionParameterFunctional('sin(10*pi*t[0])', {'t': 1}),

            dirichlet_data=ConstantFunction(value=0., dim_domain=2),

            neumann_data=ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * -1000.',
                                            dim_domain=2),
        ),

        T=1.,

        initial_data=ExpressionFunction('(x[..., 0] > 0.45) * (x[..., 0] < 0.55) * (x[..., 1] < 0.7) * 10.',
                                        dim_domain=2)
    )

    # discretize using continuous finite elements
    fom, _ = discretize_instationary_cg(analytical_problem=problem, diameter=1./GRID_INTERVALS, nt=NT)
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
    h1_mat = df.assemble(df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)
    h1_0_mat = h1_mat.copy()
    mat0 = h1_mat.copy()
    mat0.zero()
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
    bc.apply(h1_0_mat)
    bc.apply(mat0)
    bc.zero(bottom_mat)
    bc.zero(top_mat)
    bc.apply(f)
    bc.apply(u0)

    # wrap everything as a pyMOR model
    ##################################

    from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator, FenicsVisualizer

    fom = InstationaryModel(
        T=1.,

        initial_data=FenicsVectorSpace(V).make_array([u0]),

        operator=LincombOperator([FenicsMatrixOperator(mat0, V, V),
                                  FenicsMatrixOperator(h1_0_mat, V, V),
                                  FenicsMatrixOperator(bottom_mat, V, V),
                                  FenicsMatrixOperator(top_mat, V, V)],
                                 [1.,
                                  1.,
                                  100. - 1.,
                                  ExpressionParameterFunctional('top[0] - 1.', {'top': 1})]),

        rhs=VectorOperator(FenicsVectorSpace(V).make_array([f])),

        mass=FenicsMatrixOperator(l2_0_mat, V, V, name='l2'),

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


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main(BACKEND, ALG, SNAPSHOTS, RBSIZE, TEST):
    # discretize
    ############
    if BACKEND == 'pymor':
        fom = discretize_pymor()
    elif BACKEND == 'fenics':
        fom = discretize_fenics()
    else:
        raise NotImplementedError
    parameter_space=fom.parameters.space(1, 100)

    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('1.', fom.parameters)
    reductor = ParabolicRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator)

    # generate reduced model
    ########################
    if ALG == 'greedy':
        rom = reduce_greedy(fom, reductor, parameter_space, SNAPSHOTS, RBSIZE)
    elif ALG == 'adaptive_greedy':
        rom = reduce_adaptive_greedy(fom, reductor, parameter_space, SNAPSHOTS, RBSIZE)
    elif ALG == 'pod':
        rom = reduce_pod(fom, reductor, parameter_space, SNAPSHOTS, RBSIZE)
    else:
        raise NotImplementedError

    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(
        rom, fom=fom, reductor=reductor, estimator=True,
        error_norms=[lambda U: DT * np.sqrt(np.sum(fom.h1_0_semi_norm(U)[1:]**2))],
        error_norm_names=['l^2-h^1'],
        condition=False, test_mus=parameter_space.sample_randomly(TEST, seed=999), plot=True
    )

    # show results
    ##############
    print(results['summary'])
    import matplotlib.pyplot as plt
    plt.show(results['figure'])

    # write results to disk
    #######################
    from pymor.core.pickle import dump
    dump(rom, open('reduced_model.out', 'wb'))
    results.pop('figure')  # matplotlib figures cannot be serialized
    dump(results, open('results.out', 'wb'))

    # visualize reduction error for worst-approximated mu
    #####################################################
    mumax = results['max_error_mus'][0, -1]
    U = fom.solve(mumax)
    U_RB = reductor.reconstruct(rom.solve(mumax))
    if BACKEND == 'fenics':  # right now the fenics visualizer does not support time trajectories
        U = U[len(U) - 1].copy()
        U_RB = U_RB[len(U_RB) - 1].copy()
    fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                separate_colorbars=True)

    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    BACKEND, ALG, SNAPSHOTS, RBSIZE, TEST = sys.argv[1:]
    BACKEND, ALG, SNAPSHOTS, RBSIZE, TEST = BACKEND.lower(), ALG.lower(), int(SNAPSHOTS), int(RBSIZE), int(TEST)
    main(BACKEND, ALG, SNAPSHOTS, RBSIZE, TEST)
