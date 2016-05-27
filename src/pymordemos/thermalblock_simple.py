#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo.

Usage:
  thermalblock_simple.py [options] MODEL ALG SNAPSHOTS RBSIZE TEST

Arguments:
  MODEL      High-dimensional model (pymor, fenics).

  ALG        The model reduction algorithm to use
             (naive, greedy, adaptive_greedy, pod).

  SNAPSHOTS  naive:           ignored
             greedy/pod:      Number of training_set parameters per block
                              (in total SNAPSHOTS^(XBLOCKS * YBLOCKS)
                              parameters).
             adaptive_greedy: size of validation set.

  RBSIZE     Size of the reduced basis.

  TEST       Number of parameters for stochastic error estimation.
"""

from pymor.basic import *        # most common pyMOR functions and classes
from functools import partial    # fix parameters of given function


# parameters for high-dimensional models
XBLOCKS = 2
YBLOCKS = 2
GRID_INTERVALS = 100
FENICS_ORDER = 2


####################################################################################################
# High-dimensional models                                                                          #
####################################################################################################


def discretize_pymor():

    # setup analytical problem
    problem = ThermalBlockProblem(num_blocks=(XBLOCKS, YBLOCKS))

    # discretize using continuous finite elements
    d, _ = discretize_elliptic_cg(problem, diameter=1. / GRID_INTERVALS)

    return d


def discretize_fenics():
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.discretizations.mpi import mpi_wrap_discretization
        return mpi_wrap_discretization(_discretize_fenics, use_with=True, pickle_subtypes=False)
    else:
        return _discretize_fenics()


def _discretize_fenics():

    # assemble system matrices - FEniCS code
    ########################################

    import dolfin as df

    mesh = df.UnitSquareMesh(GRID_INTERVALS, GRID_INTERVALS, 'crossed')
    V = df.FunctionSpace(mesh, 'Lagrange', FENICS_ORDER)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    diffusion = df.Expression('(lower0 <= x[0]) * (open0 ? (x[0] < upper0) : (x[0] <= upper0)) *' +
                              '(lower1 <= x[1]) * (open1 ? (x[1] < upper1) : (x[1] <= upper1))',
                              lower0=0., upper0=0., open0=0,
                              lower1=0., upper1=0., open1=0,
                              element=df.FunctionSpace(mesh, 'DG', 0).ufl_element())

    def assemble_matrix(x, y, nx, ny):
        diffusion.user_parameters['lower0'] = x/nx
        diffusion.user_parameters['lower1'] = y/ny
        diffusion.user_parameters['upper0'] = (x + 1)/nx
        diffusion.user_parameters['upper1'] = (y + 1)/ny
        diffusion.user_parameters['open0'] = (x + 1 == nx)
        diffusion.user_parameters['open1'] = (y + 1 == ny)
        return df.assemble(df.inner(diffusion * df.nabla_grad(u), df.nabla_grad(v)) * df.dx)

    mats = [assemble_matrix(x, y, XBLOCKS, YBLOCKS)
            for x in range(XBLOCKS) for y in range(YBLOCKS)]
    mat0 = mats[0].copy()
    mat0.zero()
    h1_mat = df.assemble(df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)

    f = df.Constant(1.) * v * df.dx
    F = df.assemble(f)

    bc = df.DirichletBC(V, 0., df.DomainBoundary())
    for m in mats:
        bc.zero(m)
    bc.apply(mat0)
    bc.apply(h1_mat)
    bc.apply(F)

    # wrap everything as a pyMOR discretization
    ###########################################

    # FEniCS wrappers
    from pymor.gui.fenics import FenicsVisualizer
    from pymor.operators.fenics import FenicsMatrixOperator
    from pymor.vectorarrays.fenics import FenicsVector

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    parameter_functionals = [ProjectionParameterFunctional(component_name='diffusion',
                                                           component_shape=(YBLOCKS, XBLOCKS),
                                                           coordinates=(YBLOCKS - y - 1, x))
                             for x in range(XBLOCKS) for y in range(YBLOCKS)]

    # wrap operators
    ops = [FenicsMatrixOperator(mat0, V, V)] + [FenicsMatrixOperator(m, V, V) for m in mats]
    op = LincombOperator(ops, [1.] + parameter_functionals)
    rhs = VectorFunctional(ListVectorArray([FenicsVector(F, V)]))
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1_0_semi')

    # build discretization
    visualizer = FenicsVisualizer(V)
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1_0_semi': h1_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    return d


####################################################################################################
# Reduction algorithms                                                                             #
####################################################################################################


def reduce_naive(d, reductor, basis_size):

    training_set = d.parameter_space.sample_randomly(basis_size)

    snapshots = d.operator.source.empty()
    for mu in training_set:
        snapshots.append(d.solve(mu))

    rd, rc, _ = reductor(d, snapshots)

    return rd, rc


def reduce_greedy(d, reductor, snapshots, basis_size):

    training_set = d.parameter_space.sample_uniformly(snapshots)
    extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_0_semi_product)
    pool = new_parallel_pool()

    greedy_data = greedy(d, reductor, training_set,
                         extension_algorithm=extension_algorithm, max_extensions=basis_size,
                         pool=pool)

    return greedy_data['reduced_discretization'], greedy_data['reconstructor']


def reduce_adaptive_greedy(d, reductor, validation_mus, basis_size):

    extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_0_semi_product)
    pool = new_parallel_pool()

    greedy_data = adaptive_greedy(d, reductor, validation_mus=-validation_mus,
                                  extension_algorithm=extension_algorithm, max_extensions=basis_size,
                                  pool=pool)

    return greedy_data['reduced_discretization'], greedy_data['reconstructor']


def reduce_pod(d, reductor, snapshots, basis_size):

    training_set = d.parameter_space.sample_uniformly(snapshots)

    snapshots = d.operator.source.empty()
    for mu in training_set:
        snapshots.append(d.solve(mu))

    basis, singular_values = pod(snapshots, modes=basis_size, product=d.h1_0_semi_product)

    rd, rc, _ = reductor(d, basis)

    return rd, rc


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main():
    # command line argument parsing
    ###############################
    import sys
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    MODEL, ALG, SNAPSHOTS, RBSIZE, TEST = sys.argv[1:]
    MODEL, ALG, SNAPSHOTS, RBSIZE, TEST = MODEL.lower(), ALG.lower(), int(SNAPSHOTS), int(RBSIZE), int(TEST)


    # discretize
    ############
    if MODEL == 'pymor':
        d = discretize_pymor()
    elif MODEL == 'fenics':
        d = discretize_fenics()
    else:
        raise NotImplementedError


    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', d.parameter_type)
    reductor = partial(reduce_coercive,
                       error_product=d.h1_0_semi_product, coercivity_estimator=coercivity_estimator)


    # generate reduced model
    ########################
    if ALG == 'naive':
        rd, rc = reduce_naive(d, reductor, RBSIZE)
    elif ALG == 'greedy':
        rd, rc = reduce_greedy(d, reductor, SNAPSHOTS, RBSIZE)
    elif ALG == 'adaptive_greedy':
        rd, rc = reduce_adaptive_greedy(d, reductor, SNAPSHOTS, RBSIZE)
    elif ALG == 'pod':
        rd, rc = reduce_pod(d, reductor, SNAPSHOTS, RBSIZE)
    else:
        raise NotImplementedError


    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(rd, discretization=d, reconstructor=rc, estimator=True,
                                       error_norms=[d.h1_0_semi_norm], condition=True,
                                       test_mus=TEST, random_seed=999, plot=True)


    # show results
    ##############
    print(results['summary'])
    import matplotlib.pyplot
    matplotlib.pyplot.show(results['figure'])


    # write results to disk
    #######################
    from pymor.core.pickle import dump
    dump(rd, open('reduced_model.out', 'wb'))
    results.pop('figure')  # matplotlib figures cannot be serialized
    dump(results, open('results.out', 'wb'))


    # visualize reduction error for worst-approximated mu
    #####################################################
    mumax = results['max_error_mus'][0, -1]
    U = d.solve(mumax)
    U_RB = rc.reconstruct(rd.solve(mumax))
    d.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                separate_colorbars=True, block=True)


if __name__ == '__main__':
    main()
