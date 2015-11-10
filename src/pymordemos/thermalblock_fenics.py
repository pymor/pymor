#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""2D Thermalblock demo using FENICS.

Usage:
  thermalblock_fenics.py [options] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE
  thermalblock_fenics.py -h | --help


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  -h, --help             Show this message.

  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: h1].

  --without-estimator    Do not use error estimator for basis generation.

  --extension-alg=ALG    Basis extension algorithm (trivial, gram_schmidt, h1_gram_schmidt)
                         to be used [default: h1_gram_schmidt].

  --grid=NI              Use grid with 2*NI*NI elements [default: 100].

  --order=ORDER          Polynomial order of the Lagrange finite elements to use [default: 1].

  --pickle=PREFIX        Pickle reduced discretizaion, as well as reconstructor and high-dimensional
                         discretization to files with this prefix.

  -p, --plot-err         Plot error.

  --plot-solutions       Plot some example solutions.

  --reductor=RED         Reductor (error estimator) to choose (traditional, residual_basis)
                         [default: residual_basis]

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
"""

from __future__ import absolute_import, division, print_function

import sys
import time
from functools import partial
from itertools import product

from docopt import docopt
import numpy as np

from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension
from pymor.algorithms.greedy import greedy
from pymor.core.pickle import dump
from pymor.discretizations.basic import StationaryDiscretization
from pymor.operators.constructions import VectorFunctional, LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.reductors.linear import reduce_stationary_affine_linear
from pymor.reductors.stationary import reduce_stationary_coercive
from pymor.vectorarrays.list import ListVectorArray


def discretize(args):
    # first assemble all matrices for the affine decomposition
    import dolfin as df
    mesh = df.UnitSquareMesh(args['--grid'], args['--grid'], 'crossed')
    V = df.FunctionSpace(mesh, 'Lagrange', args['--order'])
    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    diffusion = df.Expression(  '(lower0 <= x[0]) * (open0 ? (x[0] < upper0) : (x[0] <= upper0)) *'
                              + '(lower1 <= x[1]) * (open1 ? (x[1] < upper1) : (x[1] <= upper1))',
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

    mats = [assemble_matrix(x, y, args['XBLOCKS'], args['YBLOCKS'])
            for x in range(args['XBLOCKS']) for y in range(args['YBLOCKS'])]
    mat0 = mats[0].copy()
    mat0.zero()
    h1_mat = df.assemble((df.inner(df.nabla_grad(u), df.nabla_grad(v)) + u * v) * df.dx)

    f = df.Constant(1.) * v * df.dx
    F = df.assemble(f)

    bc = df.DirichletBC(V, 0., df.DomainBoundary())
    for m in mats:
        bc.zero(m)
    bc.apply(mat0)
    bc.apply(F)

    # wrap everything as a pyMOR discretization
    from pymor.gui.fenics import FenicsVisualizer
    from pymor.operators.fenics import FenicsMatrixOperator
    from pymor.vectorarrays.fenics import FenicsVector
    ops = [FenicsMatrixOperator(mat0)] + [FenicsMatrixOperator(m) for m in mats]

    def parameter_functional_factory(x, y):
        return ProjectionParameterFunctional(component_name='diffusion',
                                             component_shape=(args['XBLOCKS'], args['YBLOCKS']),
                                             coordinates=(args['YBLOCKS'] - y - 1, x),
                                             name='diffusion_{}_{}'.format(x, y))
    parameter_functionals = tuple(parameter_functional_factory(x, y)
                                  for x, y in product(xrange(args['XBLOCKS']), xrange(args['YBLOCKS'])))

    op = LincombOperator(ops, (1.,) + parameter_functionals)
    rhs = VectorFunctional(ListVectorArray([FenicsVector(F)]))
    h1_product = FenicsMatrixOperator(h1_mat)
    visualizer = FenicsVisualizer(V)
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1': h1_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    return d


def thermalblock_demo(args):
    args['XBLOCKS'] = int(args['XBLOCKS'])
    args['YBLOCKS'] = int(args['YBLOCKS'])
    args['--grid'] = int(args['--grid'])
    args['--order'] = int(args['--order'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt', 'h1_gram_schmidt'}
    args['--reductor'] = args['--reductor'].lower()
    assert args['--reductor'] in {'traditional', 'residual_basis'}

    print('Discretize ...')
    discretization = discretize(args)

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        discretization.visualize(Us, legend=legend, title='Detailed Solutions for different parameters')

    print('RB generation ...')

    error_product = discretization.h1_product if args['--estimator-norm'] == 'h1' else None
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', discretization.parameter_type)
    reductors = {'residual_basis': partial(reduce_stationary_coercive, error_product=error_product,
                                           coercivity_estimator=coercivity_estimator),
                 'traditional': partial(reduce_stationary_affine_linear, error_product=error_product,
                                        coercivity_estimator=coercivity_estimator)}
    reductor = reductors[args['--reductor']]
    extension_algorithms = {'trivial': trivial_basis_extension,
                            'gram_schmidt': gram_schmidt_basis_extension,
                            'h1_gram_schmidt': partial(gram_schmidt_basis_extension, product=discretization.h1_product)}
    extension_algorithm = extension_algorithms[args['--extension-alg']]
    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         use_estimator=not args['--without-estimator'], error_norm=discretization.h1_norm,
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']

    if args['--pickle']:
        print('\nWriting reduced discretization to file {} ...'.format(args['--pickle'] + '_reduced'))
        with open(args['--pickle'] + '_reduced', 'w') as f:
            dump(rb_discretization, f)

    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()

    real_rb_size = len(greedy_data['basis'])

    mus = list(discretization.parameter_space.sample_randomly(args['--test']))

    h1_err_max = -1
    h1_est_max = -1
    cond_max = -1
    for mu in mus:
        print('.', end='')
        sys.stdout.flush()
        u = rb_discretization.solve(mu)
        URB = reconstructor.reconstruct(u)
        U = discretization.solve(mu)
        h1_err = discretization.h1_norm(U - URB)[0]
        h1_est = rb_discretization.estimate(u, mu=mu)
        cond = np.linalg.cond(rb_discretization.operator.assemble(mu)._matrix)
        if h1_err > h1_err_max:
            h1_err_max = h1_err
            mumax = mu
        if h1_est > h1_est_max:
            h1_est_max = h1_est
            mu_est_max = mu
        if cond > cond_max:
            cond_max = cond
            cond_max_mu = mu
    print()

    toc = time.time()
    t_est = toc - tic

    print('''
    *** RESULTS ***

    Problem:
       number of blocks:                   {args[XBLOCKS]}x{args[YBLOCKS]}
       h:                                  sqrt(2)/{args[--grid]}

    Greedy basis generation:
       number of snapshots:                {args[SNAPSHOTS]}^({args[XBLOCKS]}x{args[YBLOCKS]})
       estimator disabled:                 {args[--without-estimator]}
       estimator norm:                     {args[--estimator-norm]}
       extension method:                   {args[--extension-alg]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal H1-error:                   {h1_err_max}  (mu = {mumax})
       maximal condition of system matrix: {cond_max}  (mu = {cond_max_mu})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()

    if args['--plot-err']:
        U = discretization.solve(mumax)
        URB = reconstructor.reconstruct(rb_discretization.solve(mumax))
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution')


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
