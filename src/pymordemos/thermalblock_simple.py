#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Simplified version of the thermalblock demo.

Usage:
  thermalblock_simple.py MODEL ALG SNAPSHOTS RBSIZE TEST

Arguments:
"""

from typer import Argument, run

from pymor.basic import *
from pymor.tools.typer import Choices
from pymor.core.config import config


# parameters for high-dimensional models
XBLOCKS = 2             # pyMOR/FEniCS
YBLOCKS = 2             # pyMOR/FEniCS
GRID_INTERVALS = 100    # pyMOR/FEniCS
FENICS_ORDER = 2
NGS_ORDER = 4
TEXT = 'pyMOR'


####################################################################################################
# Main script                                                                                      #
####################################################################################################

def main(
    model: Choices('pymor fenics ngsolve pymor_text') = Argument(..., help='High-dimensional model.'),
    alg: Choices('naive greedy adaptive_greedy pod') = Argument(..., help='The model reduction algorithm to use.'),
    snapshots: int = Argument(
        ...,
        help='naive: ignored.\n\n'
             'greedy/pod: Number of training_set parameters per block'
             '(in total SNAPSHOTS^(XBLOCKS * YBLOCKS) parameters).\n\n'
             'adaptive_greedy: size of validation set.'
    ),
    rbsize: int = Argument(..., help='Size of the reduced basis.'),
    test: int = Argument(..., help='Number of parameters for stochastic error estimation.'),
):
    # discretize
    ############
    if model == 'pymor':
        fom, parameter_space = discretize_pymor()
    elif model == 'fenics':
        fom, parameter_space = discretize_fenics()
    elif model == 'ngsolve':
        config.require("NGSOLVE")
        fom, parameter_space = discretize_ngsolve()
    elif model == 'pymor_text':
        fom, parameter_space = discretize_pymor_text()
    else:
        raise NotImplementedError

    # select reduction algorithm with error estimator
    #################################################
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', fom.parameters)
    reductor = CoerciveRBReductor(fom, product=fom.h1_0_semi_product, coercivity_estimator=coercivity_estimator,
                                  check_orthonormality=False)

    # generate reduced model
    ########################
    if alg == 'naive':
        rom = reduce_naive(fom, reductor, parameter_space, rbsize)
    elif alg == 'greedy':
        rom = reduce_greedy(fom, reductor, parameter_space, snapshots, rbsize)
    elif alg == 'adaptive_greedy':
        rom = reduce_adaptive_greedy(fom, reductor, parameter_space, snapshots, rbsize)
    elif alg == 'pod':
        rom = reduce_pod(fom, reductor, parameter_space, snapshots, rbsize)
    else:
        raise NotImplementedError

    # evaluate the reduction error
    ##############################
    results = reduction_error_analysis(rom, fom=fom, reductor=reductor, error_estimator=True,
                                       error_norms=[fom.h1_0_semi_norm], condition=True,
                                       test_mus=parameter_space.sample_randomly(test))

    # show results
    ##############
    print(results['summary'])
    plot_reduction_error_analysis(results)

    # write results to disk
    #######################
    from pymor.core.pickle import dump
    dump((rom, parameter_space), open('reduced_model.out', 'wb'))
    dump(results, open('results.out', 'wb'))

    # visualize reduction error for worst-approximated mu
    #####################################################
    mumax = results['max_error_mus'][0, -1]
    U = fom.solve(mumax)
    U_RB = reductor.reconstruct(rom.solve(mumax))
    fom.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                  separate_colorbars=True, block=True)


####################################################################################################
# High-dimensional models                                                                          #
####################################################################################################


def discretize_pymor():

    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(XBLOCKS, YBLOCKS))

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1. / GRID_INTERVALS)

    return fom, problem.parameter_space


def discretize_fenics():
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        fom = mpi_wrap_model(_discretize_fenics, use_with=True, pickle_local_spaces=False)
    else:
        fom = _discretize_fenics()
    return fom, fom.parameters.space((0.1, 1))


def _discretize_fenics():

    # assemble system matrices - FEniCS code
    ########################################

    import dolfin as df

    mesh = df.UnitSquareMesh(GRID_INTERVALS, GRID_INTERVALS, 'crossed')
    V = df.FunctionSpace(mesh, 'Lagrange', FENICS_ORDER)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    diffusion = df.Expression('(lower0 <= x[0]) * (open0 ? (x[0] < upper0) : (x[0] <= upper0)) *'
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

    # wrap everything as a pyMOR model
    ##################################

    # FEniCS wrappers
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator, FenicsVisualizer

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    parameter_functionals = [ProjectionParameterFunctional('diffusion',
                                                           size=YBLOCKS*XBLOCKS,
                                                           index=YBLOCKS - y - 1 + x*YBLOCKS)
                             for x in range(XBLOCKS) for y in range(YBLOCKS)]

    # wrap operators
    ops = [FenicsMatrixOperator(mat0, V, V)] + [FenicsMatrixOperator(m, V, V) for m in mats]
    op = LincombOperator(ops, [1.] + parameter_functionals)
    rhs = VectorOperator(FenicsVectorSpace(V).make_array([F]))
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1_0_semi')

    # build model
    visualizer = FenicsVisualizer(FenicsVectorSpace(V))
    fom = StationaryModel(op, rhs, products={'h1_0_semi': h1_product},
                          visualizer=visualizer)

    return fom


def discretize_ngsolve():
    from ngsolve import (ngsglobals, Mesh, H1, CoefficientFunction, LinearForm, SymbolicLFI,
                         BilinearForm, SymbolicBFI, grad, TaskManager)
    from netgen.csg import CSGeometry, OrthoBrick, Pnt
    import numpy as np

    ngsglobals.msg_level = 1

    geo = CSGeometry()
    obox = OrthoBrick(Pnt(-1, -1, -1), Pnt(1, 1, 1)).bc("outer")

    b = []
    b.append(OrthoBrick(Pnt(-1, -1, -1), Pnt(0.0, 0.0, 0.0)).mat("mat1").bc("inner"))
    b.append(OrthoBrick(Pnt(-1,  0, -1), Pnt(0.0, 1.0, 0.0)).mat("mat2").bc("inner"))
    b.append(OrthoBrick(Pnt(0,  -1, -1), Pnt(1.0, 0.0, 0.0)).mat("mat3").bc("inner"))
    b.append(OrthoBrick(Pnt(0,   0, -1), Pnt(1.0, 1.0, 0.0)).mat("mat4").bc("inner"))
    b.append(OrthoBrick(Pnt(-1, -1,  0), Pnt(0.0, 0.0, 1.0)).mat("mat5").bc("inner"))
    b.append(OrthoBrick(Pnt(-1,  0,  0), Pnt(0.0, 1.0, 1.0)).mat("mat6").bc("inner"))
    b.append(OrthoBrick(Pnt(0,  -1,  0), Pnt(1.0, 0.0, 1.0)).mat("mat7").bc("inner"))
    b.append(OrthoBrick(Pnt(0,   0,  0), Pnt(1.0, 1.0, 1.0)).mat("mat8").bc("inner"))
    box = (obox - b[0] - b[1] - b[2] - b[3] - b[4] - b[5] - b[6] - b[7])

    geo.Add(box)
    for bi in b:
        geo.Add(bi)
    # domain 0 is empty!

    mesh = Mesh(geo.GenerateMesh(maxh=0.3))

    # H1-conforming finite element space
    V = H1(mesh, order=NGS_ORDER, dirichlet="outer")
    v = V.TestFunction()
    u = V.TrialFunction()

    # Coeff as array: variable coefficient function (one CoefFct. per domain):
    sourcefct = CoefficientFunction([1 for i in range(9)])

    with TaskManager():
        # the right hand side
        f = LinearForm(V)
        f += SymbolicLFI(sourcefct * v)
        f.Assemble()

        # the bilinear-form
        mats = []
        coeffs = [[0, 1, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1, 0, 0, 0]]
        for c in coeffs:
            diffusion = CoefficientFunction(c)
            a = BilinearForm(V, symmetric=False)
            a += SymbolicBFI(diffusion * grad(u) * grad(v), definedon=(np.where(np.array(c) == 1)[0] + 1).tolist())
            a.Assemble()
            mats.append(a.mat)

    from pymor.bindings.ngsolve import NGSolveVectorSpace, NGSolveMatrixOperator, NGSolveVisualizer

    space = NGSolveVectorSpace(V)
    op = LincombOperator([NGSolveMatrixOperator(m, space, space) for m in mats],
                         [ProjectionParameterFunctional('diffusion', len(coeffs), i) for i in range(len(coeffs))])

    h1_0_op = op.assemble(Mu(diffusion=[1] * len(coeffs))).with_(name='h1_0_semi')

    F = space.zeros()
    F.vectors[0].real_part.impl.vec.data = f.vec
    F = VectorOperator(F)

    fom = StationaryModel(op, F, visualizer=NGSolveVisualizer(mesh, V),
                          products={'h1_0_semi': h1_0_op})

    return fom, fom.parameters.space((0.1, 1))


def discretize_pymor_text():

    # setup analytical problem
    problem = text_problem(TEXT)

    # discretize using continuous finite elements
    fom, _ = discretize_stationary_cg(problem, diameter=1.)

    return fom, problem.parameter_space


####################################################################################################
# Reduction algorithms                                                                             #
####################################################################################################


def reduce_naive(fom, reductor, parameter_space, basis_size):

    training_set = parameter_space.sample_randomly(basis_size)

    for mu in training_set:
        reductor.extend_basis(fom.solve(mu), method='trivial')

    rom = reductor.reduce()

    return rom


def reduce_greedy(fom, reductor, parameter_space, snapshots, basis_size):

    training_set = parameter_space.sample_uniformly(snapshots)
    pool = new_parallel_pool()

    greedy_data = rb_greedy(fom, reductor, training_set,
                            extension_params={'method': 'gram_schmidt'},
                            max_extensions=basis_size,
                            pool=pool)

    return greedy_data['rom']


def reduce_adaptive_greedy(fom, reductor, parameter_space, validation_mus, basis_size):

    pool = new_parallel_pool()

    greedy_data = rb_adaptive_greedy(fom, reductor, parameter_space,
                                     validation_mus=-validation_mus,
                                     extension_params={'method': 'gram_schmidt'},
                                     max_extensions=basis_size,
                                     pool=pool)

    return greedy_data['rom']


def reduce_pod(fom, reductor, parameter_space, snapshots, basis_size):

    training_set = parameter_space.sample_uniformly(snapshots)

    snapshots = fom.operator.source.empty()
    for mu in training_set:
        snapshots.append(fom.solve(mu))

    basis, singular_values = pod(snapshots, modes=basis_size, product=reductor.products['RB'])
    reductor.extend_basis(basis, method='trivial')

    rom = reductor.reduce()

    return rom


if __name__ == '__main__':
    run(main)
