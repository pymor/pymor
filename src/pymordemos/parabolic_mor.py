#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Parabolic problem demo.

Usage:
  parabolic_mor.py [options] MODEL ALG REDUCTOR GAMMA SNAPSHOTS RBSIZE TEST

Arguments:
  MODEL      High-dimensional model (pymor, fenics).

  ALG        The model reduction algorithm to use
             (naive, greedy, adaptive_greedy, pod).

  REDUCTOR   The reductor and corresponding error estimator to use
             (l2_estimate, l2_estimate_simple, energy_estimate, energy_estimate_simple)

  GAMMA      Value of gamma for energy estimate.

  SNAPSHOTS  naive:           ignored
             greedy/pod:      Number of training_set parameters
             adaptive_greedy: size of validation set.

  RBSIZE     Size of the reduced basis.

  TEST       Number of parameters for stochastic error estimation.
"""

from __future__ import division  # ensure that 1 / 2 is 0.5 and not 0
from pymor.basic import *        # most common pyMOR functions and classes
from pymor.analyticalproblems.parabolic import ParabolicProblem
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper
from pymor.discretizers.parabolic import discretize_parabolic_cg
from pymor.reductors.parabolic import reduce_parabolic_l2_estimate, reduce_parabolic_l2_estimate_simple, \
    reduce_parabolic_energy_estimate, reduce_parabolic_energy_estimate_simple
import numpy as np
from functools import partial    # fix parameters of given function


# parameters for high-dimensional models
GRID_INTERVALS = 100
FENICS_ORDER = 1


####################################################################################################
# High-dimensional models                                                                          #
####################################################################################################


def discretize_pymor():

    # setup analytical problem
    domain = RectDomain(top=BoundaryType('neumann'), bottom=BoundaryType('neumann'))
    rhs = ConstantFunction(value=0., dim_domain=2)
    diffusion_functional = GenericParameterFunctional(mapping=lambda mu: mu['diffusion'],
                                                      parameter_type={'diffusion': 0})
    dirichlet = ConstantFunction(value=0., dim_domain=2)
    neumann = ConstantFunction(value=-1., dim_domain=2)
    initial = GenericFunction(lambda X: np.cos(np.pi*X[..., 0])*np.sin(np.pi*X[..., 1]), dim_domain=2)

    problem = ParabolicProblem(domain=domain, rhs=rhs, diffusion_functionals=[diffusion_functional],
                               dirichlet_data=dirichlet, neumann_data=neumann, initial_data=initial,
                               parameter_space=CubicParameterSpace({'diffusion': 0}, minimum=0.1, maximum=1.))

    # discretize using continuous finite elements
    grid, bi = discretize_domain_default(problem.domain, diameter=1. / GRID_INTERVALS, grid_type=TriaGrid)
    d, _ = discretize_parabolic_cg(analytical_problem=problem, grid=grid, boundary_info=bi, nt=100)

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

    l2_mat = df.assemble(df.inner(u, v) * df.dx)
    l2_0_mat = l2_mat.copy()
    h1_mat = df.assemble(df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)
    h1_0_mat = h1_mat.copy()
    mat0 = h1_mat.copy()
    mat0.zero()

    f = df.Constant(0.) * v * df.dx + df.Constant(1.) * v * df.ds
    F = df.assemble(f)

    def dirichlet_boundary(x, on_boundary):
        tol = 1e-14
        return on_boundary and (abs(x[0]) < tol or abs(x[0] - 1) < tol)

    bc_dirichlet = df.DirichletBC(V, df.Constant(0.), dirichlet_boundary)
    bc_dirichlet.apply(l2_0_mat)
    bc_dirichlet.apply(h1_0_mat)
    bc_dirichlet.apply(mat0)
    bc_dirichlet.apply(F)

    initial = df.project(df.Expression('cos(pi*x[0]) * sin(pi*x[1])'), V).vector()

    # wrap everything as a pyMOR discretization
    ###########################################

    # FEniCS wrappers
    from pymor.gui.fenics import FenicsVisualizer
    from pymor.operators.fenics import FenicsMatrixOperator
    from pymor.vectorarrays.fenics import FenicsVector

    parameter_functional = GenericParameterFunctional(mapping=lambda mu: mu['diffusion'],
                                                      parameter_type={'diffusion': 0})

    # wrap operators
    initial_data = ListVectorArray([FenicsVector(initial, V)])
    op = LincombOperator([FenicsMatrixOperator(mat0, V, V), FenicsMatrixOperator(h1_0_mat, V, V)],
                         [1., parameter_functional])
    rhs = VectorFunctional(ListVectorArray([FenicsVector(F, V)]))
    l2_product = FenicsMatrixOperator(l2_mat, V, V, name='l2')
    l2_0_product = FenicsMatrixOperator(l2_0_mat, V, V, name='l2_0')
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1')
    h1_0_product = FenicsMatrixOperator(h1_0_mat, V, V, name='h1_0_semi')

    # build discretization
    time_stepper = ImplicitEulerTimeStepper(nt=100)
    visualizer = FenicsVisualizer(V)
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = InstationaryDiscretization(1., initial_data=initial_data, operator=op, rhs=rhs, mass=l2_0_product,
                                   time_stepper=time_stepper,
                                   products={'l2': l2_product, 'l2_0': l2_0_product,
                                             'h1': h1_product, 'h1_0_semi': h1_0_product},
                                   parameter_space=parameter_space, visualizer=visualizer)

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
    extension_algorithm = pod_basis_extension
    pool = new_parallel_pool()

    greedy_data = greedy(d, reductor, training_set,
                         extension_algorithm=extension_algorithm, max_extensions=basis_size,
                         pool=pool)

    return greedy_data['reduced_discretization'], greedy_data['reconstructor']


def reduce_adaptive_greedy(d, reductor, validation_mus, basis_size):

    extension_algorithm = pod_basis_extension
    pool = new_parallel_pool()

    greedy_data = adaptive_greedy(d, reductor, validation_mus=validation_mus,
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
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(1)
    MODEL, ALG, REDUCTOR, GAMMA, SNAPSHOTS, RBSIZE, TEST = sys.argv[1:]
    MODEL, ALG, REDUCTOR, GAMMA, SNAPSHOTS, RBSIZE, TEST = MODEL.lower(), ALG.lower(), REDUCTOR.lower(), float(GAMMA), \
                                                           int(SNAPSHOTS), int(RBSIZE), int(TEST)


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
    reductors = {'l2_estimate': reduce_parabolic_l2_estimate,
                 'l2_estimate_simple': reduce_parabolic_l2_estimate_simple,
                 'energy_estimate': partial(reduce_parabolic_energy_estimate, error_product=d.h1_0_semi_product,
                                            coercivity_estimator=coercivity_estimator, gamma=GAMMA),
                 'energy_estimate_simple': partial(reduce_parabolic_energy_estimate_simple,
                                                   error_product=d.h1_0_semi_product,
                                                   coercivity_estimator=coercivity_estimator, gamma=GAMMA)}
    if REDUCTOR in ('l2_estimate', 'l2_estimate_simple', 'energy_estimate', 'energy_estimate_simple'):
        reductors = {REDUCTOR: reductors[REDUCTOR]}
    elif REDUCTOR == 'all':
        reductors = reductors
    else:
        raise NotImplementedError

    results = {}
    for reductor_name, reductor in reductors.iteritems():
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
        results[reductor_name] = reduction_error_analysis(rd, discretization=d, reconstructor=rc, estimator=True,
                                           error_norms=[lambda U: np.max(d.l2_norm(U))], error_norm_names=['error'],
                                           condition=False, test_mus=TEST, random_seed=999, plot=True)


        # show results
        ##############
        print(results[reductor_name]['summary'])
        import matplotlib.pyplot as plt
        plt.show(results[reductor_name]['figure'])


        # write results to disk
        #######################
        from pymor.core.pickle import dump
        dump(rd, open('reduced_model_'+reductor_name+'.out', 'wb'))
        results[reductor_name].pop('figure')  # matplotlib figures cannot be serialized
        dump(results[reductor_name], open('results_'+reductor_name+'.out', 'wb'))


        # visualize reduction error for worst-approximated mu
        #####################################################
        mumax = results[reductor_name]['max_error_mus'][0, -1]
        U = d.solve(mumax)
        U_RB = rc.reconstruct(rd.solve(mumax))
        d.visualize((U, U_RB, U - U_RB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                    separate_colorbars=True, block=False)

    if len(reductors) > 1:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        legend = []
        for reductor_name, reductor in reductors.iteritems():
            ax.semilogy(results[reductor_name]['basis_sizes'], results[reductor_name]['max_errors'][0])
            legend.append('error: '+reductor_name)
            ax.semilogy(results[reductor_name]['basis_sizes'], results[reductor_name]['max_estimates'], ls='--')
            legend.append('estimator: '+reductor_name)
        ax.legend(legend)
        ax.set_title('maximum errors')
        plt.show(fig)


if __name__ == '__main__':
    main()
