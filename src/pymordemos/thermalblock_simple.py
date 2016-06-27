#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo.

Usage:
  thermalblock_simple.py [options] MODEL ALG SNAPSHOTS RBSIZE TEST

Arguments:
  MODEL      High-dimensional model (pymor, fenics, ngsolve).

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
    problem = thermal_block_problem(num_blocks=(XBLOCKS, YBLOCKS))

    # discretize using continuous finite elements
    d, _ = discretize_stationary_cg(problem, diameter=1. / GRID_INTERVALS)

    return d


def discretize_fenics():
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.discretizations.mpi import mpi_wrap_discretization
        return mpi_wrap_discretization(_discretize_fenics, use_with=True, pickle_local_spaces=False)
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
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator, FenicsVisualizer

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    parameter_functionals = [ProjectionParameterFunctional(component_name='diffusion',
                                                           component_shape=(YBLOCKS, XBLOCKS),
                                                           coordinates=(YBLOCKS - y - 1, x))
                             for x in range(XBLOCKS) for y in range(YBLOCKS)]

    # wrap operators
    ops = [FenicsMatrixOperator(mat0, V, V)] + [FenicsMatrixOperator(m, V, V) for m in mats]
    op = LincombOperator(ops, [1.] + parameter_functionals)
    rhs = VectorFunctional(FenicsVectorSpace(V).make_array([F]))
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1_0_semi')

    # build discretization
    visualizer = FenicsVisualizer(FenicsVectorSpace(V))
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1_0_semi': h1_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    return d


def discretize_ngsolve():
    from ngsolve import (ngsglobals, Mesh, H1, CoefficientFunction, LinearForm, SymbolicLFI,
                         BilinearForm, SymbolicBFI, grad, TaskManager)
    from netgen.geom2d import SplineGeometry

    ngsglobals.msg_level = 1

    geo = SplineGeometry()

    #
    # domains 1,2,3,4 and boundary condition labels (sw,se,ws,...):
    # 
    #                          D0 ( outer domain )    
    #                     
    #                      6-----nw----7-----ne-----8
    #                      |           |            |
    #                      |           |            |
    # D0 ( outer domain )  w     D3    i     D4     e  D0 ( outer domain )
    #                      n           n            n
    #                      |           |            |
    #                      3-----iw----4-----ie-----5
    #                      |           |            |
    #                      |           |            |
    # D0 ( outer domain )  w     D1    i     D2     e  D0 ( outer domain )
    #                      s           s            s
    #                      |           |            |
    #                      0-----sw----1-----se-----2
    #                          D0 ( outer domain )
    #                                                                      
    pts = [geo.AppendPoint(*p) for p in [(-1, -1),
                                         (0,  -1),
                                         (1,  -1),
                                         (-1,  0),
                                         (0,   0),
                                         (1,   0),
                                         (-1,  1),
                                         (0,   1),
                                         (1,   1)]]

    geo.Append(["line", pts[0], pts[1]], bc="sw", leftdomain=1, rightdomain=0)
    geo.Append(["line", pts[1], pts[2]], bc="se", leftdomain=2, rightdomain=0)
    geo.Append(["line", pts[3], pts[4]], bc="iw", leftdomain=3, rightdomain=1)
    geo.Append(["line", pts[4], pts[5]], bc="ie", leftdomain=4, rightdomain=2)
    geo.Append(["line", pts[6], pts[7]], bc="nw", leftdomain=0, rightdomain=3)
    geo.Append(["line", pts[7], pts[8]], bc="ne", leftdomain=0, rightdomain=4)
    geo.Append(["line", pts[0], pts[3]], bc="ws", leftdomain=0, rightdomain=1)
    geo.Append(["line", pts[3], pts[6]], bc="wn", leftdomain=0, rightdomain=3)
    geo.Append(["line", pts[1], pts[4]], bc="is", leftdomain=1, rightdomain=2)
    geo.Append(["line", pts[4], pts[7]], bc="in", leftdomain=3, rightdomain=4)
    geo.Append(["line", pts[2], pts[5]], bc="es", leftdomain=2, rightdomain=0)
    geo.Append(["line", pts[5], pts[8]], bc="en", leftdomain=4, rightdomain=0)


    # generate a triangular mesh of mesh-size 0.2
    mesh = Mesh(geo.GenerateMesh(maxh=0.2))

    # H1-conforming finite element space
    V = H1(mesh, order=1, dirichlet="sw|ws|se|es|nw|wn|ne|en")
    v = V.TestFunction()
    u = V.TrialFunction()

    # domain constant coefficient function (one value per domain):
    # sourcefct = DomainConstantCF((1,1,1,1))
    # or as a coeff as array: variable coefficient function (one CoefFct. per domain):
    sourcefct = CoefficientFunction([1, 1, 1, 1])

    with TaskManager():
        # the right hand side
        f = LinearForm(V)
        f += SymbolicLFI(sourcefct * v)
        f.Assemble()


        # the bilinear-form
        mats = []
        for i in range(4):
            coeffs = [0] * 4
            coeffs[i] = 1
            diffusion = CoefficientFunction(coeffs)
            a = BilinearForm(V, symmetric=False)
            a += SymbolicBFI(diffusion * grad(u) * grad(v), definedon=[i])
            a.Assemble()
            mats.append(a.mat)

    from pymor.gui.ngsolve import NGSolveVisualizer
    from pymor.operators.ngsolve import NGSolveMatrixOperator

    op = LincombOperator([NGSolveMatrixOperator(m, V.FreeDofs()) for m in mats],
                         [ProjectionParameterFunctional('diffusion', (4,), (i,)) for i in range(4)])

    h1_0_op = op.assemble([1, 1, 1, 1])

    F = op.range.zeros()
    F._list[0].impl.data = f.vec
    F = VectorFunctional(F)

    return StationaryDiscretization(op, F, visualizer=NGSolveVisualizer(V),
                                    products={'h1_0_semi': h1_0_op},
                                    parameter_space=CubicParameterSpace(op.parameter_type, 0.1, 1.))


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
    elif MODEL == 'ngsolve':
        d = discretize_ngsolve()
    else:
        raise NotImplementedError

    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', d.parameter_type)
    reductor = partial(reduce_coercive,
                       product=d.h1_0_semi_product, coercivity_estimator=coercivity_estimator)

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
