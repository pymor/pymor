#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Thermalblock demo.

Usage:
  thermalblock.py [options] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE
  thermalblock.py -h | --help


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  -h, --help                 Show this message.

  --estimator-norm=NORM      Norm (trivial, h1) in which to calculate the residual
                             [default: h1].

  --without-estimator        Do not use error estimator for basis generation.

  --extension-alg=ALG        Basis extension algorithm (trivial, gram_schmidt, h1_gram_schmidt)
                             to be used [default: h1_gram_schmidt].

  --fenics                   Use FEniCS discretization.

  --grid=NI                  Use grid with 2*NI*NI elements [default: 100].

  --order=ORDER              Polynomial order of the Lagrange finite elements to use in FEniCS
                             discretization [default: 1].

  --pickle=PREFIX            Pickle reduced discretizaion, as well as reconstructor and high-dimensional
                             discretization to files with this prefix.

  -p, --plot-err             Plot error.

  --plot-solutions           Plot some example solutions.

  --plot-error-sequence      Plot reduction error vs. basis size.

  --reductor=RED             Reductor (error estimator) to choose (traditional, residual_basis)
                             [default: residual_basis]

  --test=COUNT               Use COUNT snapshots for stochastic error estimation
                             [default: 10].

  --ipython-engines=COUNT    If positive, the number of IPython cluster engines to use for
                             parallel greedy search. If zero, no parallelization is performed.
                             [default: 0]

  --ipython-profile=PROFILE  IPython profile to use for parallelization.

  --cache-region=REGION      Name of cache region to use for caching solution snapshots
                             (NONE, MEMORY, DISK, PERSISTENT)
                             [default: NONE]

  --list-vector-array        Solve using ListVectorArray[NumpyVector] instead of NumpyVectorArray.

  --pod                      Use POD instead of greedy algorithm for basis generation.

  --pod-product=PROD         Inner product w.r.t. with to compute the pod (trivial, h1)
                             [default: h1].
"""

from __future__ import absolute_import, division, print_function

from functools import partial
from itertools import product
import sys
import time

from docopt import docopt

from pymor.algorithms.error import reduction_error_analysis
from pymor.core.pickle import dump
from pymor.parallel.default import new_parallel_pool


def thermalblock_demo(args):

    args = parse_arguments(args)

    pool = new_parallel_pool(ipython_num_engines=args['--ipython-engines'], ipython_profile=args['--ipython-profile'])

    if args['--fenics']:
        d, d_summary = discretize_fenics(args['XBLOCKS'], args['YBLOCKS'], args['--grid'], args['--order'])
    else:
        d, d_summary = discretize_pymor(args['XBLOCKS'], args['YBLOCKS'], args['--grid'], args['--list-vector-array'])

    if args['--cache-region'] != 'none':
        d.enable_caching(args['--cache-region'])

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = tuple()
        legend = tuple()
        for mu in d.parameter_space.sample_randomly(2):
            print('Solving for diffusion = \n{} ... '.format(mu['diffusion']))
            sys.stdout.flush()
            Us = Us + (d.solve(mu),)
            legend = legend + (str(mu['diffusion']),)
        d.visualize(Us, legend=legend, title='Detailed Solutions for different parameters',
                    separate_colorbars=False, block=True)

    print('RB generation ...')

    # define estimator for coercivity constant
    from pymor.parameters.functionals import ExpressionParameterFunctional
    coercivity_estimator = ExpressionParameterFunctional('min(diffusion)', d.parameter_type)

    # inner product for computation of Riesz representatives
    error_product = d.h1_0_semi_product if args['--estimator-norm'] == 'h1' else None

    if args['--reductor'] == 'residual_basis':
        from pymor.reductors.stationary import reduce_stationary_coercive
        reductor = partial(reduce_stationary_coercive, error_product=error_product,
                           coercivity_estimator=coercivity_estimator)
    elif args['--reductor'] == 'traditional':
        from pymor.reductors.linear import reduce_stationary_affine_linear
        reductor = partial(reduce_stationary_affine_linear, error_product=error_product,
                           coercivity_estimator=coercivity_estimator)
    else:
        assert False  # this should never happen

    if args['--pod']:
        rd, rc, red_summary = reduce_pod(d=d, reductor=reductor, snapshots_per_block=args['SNAPSHOTS'],
                                         basis_size=args['RBSIZE'], product_name=args['--pod-product'])
    else:
        rd, rc, red_summary = reduce_greedy(d=d, reductor=reductor, snapshots_per_block=args['SNAPSHOTS'],
                                            extension_alg_name=args['--extension-alg'],
                                            max_extensions=args['RBSIZE'],
                                            use_estimator=not args['--without-estimator'], pool=pool)

    if args['--pickle']:
        print('\nWriting reduced discretization to file {} ...'.format(args['--pickle'] + '_reduced'))
        with open(args['--pickle'] + '_reduced', 'w') as f:
            dump(rd, f)
        if not args['--fenics']:  # FEniCS data structures do not support serialization
            print('Writing detailed discretization and reconstructor to file {} ...'
                  .format(args['--pickle'] + '_detailed'))
            with open(args['--pickle'] + '_detailed', 'w') as f:
                dump((d, rc), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rd,
                                       discretization=d,
                                       reconstructor=rc,
                                       estimator=True,
                                       error_norms=(d.h1_0_semi_norm, d.l2_norm),
                                       condition=True,
                                       test_mus=args['--test'],
                                       basis_sizes=0 if args['--plot-error-sequence'] else 1,
                                       plot=args['--plot-error-sequence'],
                                       pool=pool)

    print('\n*** RESULTS ***\n')
    print(d_summary)
    print(red_summary)
    print(results['summary'])
    sys.stdout.flush()

    if args['--plot-error-sequence']:
        from matplotlib import pyplot as plt
        plt.show(results['figure'])
    if args['--plot-err']:
        mumax = results['max_error_mus'][0, -1]
        U = d.solve(mumax)
        URB = rc.reconstruct(rd.solve(mumax))
        d.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                    title='Maximum Error Solution', separate_colorbars=True, block=True)


def parse_arguments(args):
    args['XBLOCKS'] = int(args['XBLOCKS'])
    args['YBLOCKS'] = int(args['YBLOCKS'])
    args['--grid'] = int(args['--grid'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])
    args['--test'] = int(args['--test'])
    args['--ipython-engines'] = int(args['--ipython-engines'])
    args['--estimator-norm'] = args['--estimator-norm'].lower()
    assert args['--estimator-norm'] in {'trivial', 'h1'}
    args['--extension-alg'] = args['--extension-alg'].lower()
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt', 'h1_gram_schmidt'}
    args['--reductor'] = args['--reductor'].lower()
    assert args['--reductor'] in {'traditional', 'residual_basis'}
    args['--cache-region'] = args['--cache-region'].lower()
    args['--order'] = int(args['--order'])
    return args


def discretize_pymor(xblocks, yblocks, grid_num_intervals, use_list_vector_array):
    from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
    from pymor.discretizers.elliptic import discretize_elliptic_cg
    from pymor.playground.discretizers.numpylistvectorarray import convert_to_numpy_list_vector_array

    print('Discretize ...')
    # setup analytical problem
    problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']))

    # discretize using continuous finite elements
    d, _ = discretize_elliptic_cg(problem, diameter=1. / args['--grid'])

    if use_list_vector_array:
        d = convert_to_numpy_list_vector_array(d)

    summary = '''pyMOR discretization:
   number of blocks: {xblocks}x{yblocks}
   grid intervals:   {grid_num_intervals}
   ListVectorArray:  {use_list_vector_array}
'''.format(**locals())

    return d, summary


def discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):

    # assemble system matrices - FEniCS code
    ########################################

    import dolfin as df

    mesh = df.UnitSquareMesh(grid_num_intervals, grid_num_intervals, 'crossed')
    V = df.FunctionSpace(mesh, 'Lagrange', element_order)
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

    mats = [assemble_matrix(x, y, xblocks, yblocks)
            for x in range(xblocks) for y in range(yblocks)]
    mat0 = mats[0].copy()
    mat0.zero()
    h1_mat = df.assemble(df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx)
    l2_mat = df.assemble(u * v * df.dx)

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

    # generic pyMOR classes
    from pymor.discretizations.basic import StationaryDiscretization
    from pymor.operators.constructions import LincombOperator, VectorFunctional
    from pymor.parameters.functionals import ProjectionParameterFunctional
    from pymor.parameters.spaces import CubicParameterSpace
    from pymor.vectorarrays.list import ListVectorArray

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    def parameter_functional_factory(x, y):
        return ProjectionParameterFunctional(component_name='diffusion',
                                             component_shape=(yblocks, xblocks),
                                             coordinates=(yblocks - y - 1, x),
                                             name='diffusion_{}_{}'.format(x, y))
    parameter_functionals = tuple(parameter_functional_factory(x, y)
                                  for x, y in product(xrange(args['XBLOCKS']), xrange(args['YBLOCKS'])))

    # wrap operators
    ops = [FenicsMatrixOperator(mat0, V, V)] + [FenicsMatrixOperator(m, V, V) for m in mats]
    op = LincombOperator(ops, (1.,) + parameter_functionals)
    rhs = VectorFunctional(ListVectorArray([FenicsVector(F, V)]))
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1_0_semi')
    l2_product = FenicsMatrixOperator(l2_mat, V, V, name='l2')

    # build discretization
    visualizer = FenicsVisualizer(V)
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1_0_semi': h1_product,
                                                    'l2': l2_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    summary = '''FEniCS discretization:
   number of blocks:      {xblocks}x{yblocks}
   grid intervals:        {grid_num_intervals}
   finite element order:  {element_order}
'''.format(**locals())

    return d, summary


def reduce_greedy(d, reductor, snapshots_per_block,
                  extension_alg_name, max_extensions, use_estimator, pool):

    from pymor.algorithms.basisextension import trivial_basis_extension, gram_schmidt_basis_extension
    from pymor.algorithms.greedy import greedy

    # choose basis extension algorithm
    if extension_alg_name == 'trivial':
        extension_algorithm = trivial_basis_extension
    elif extension_alg_name == 'gram_schmidt':
        extension_algorithm = gram_schmidt_basis_extension
    elif extension_alg_name == 'h1_gram_schmidt':
        extension_algorithm = partial(gram_schmidt_basis_extension, product=d.h1_0_semi_product)
    else:
        assert False

    # run greedy
    training_set = d.parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = greedy(d, reductor, training_set,
                         use_estimator=use_estimator, error_norm=d.h1_0_semi_norm,
                         extension_algorithm=extension_algorithm, max_extensions=max_extensions,
                         pool=pool)
    rd, rc = greedy_data['reduced_discretization'], greedy_data['reconstructor']

    # generate summary
    real_rb_size = rd.solution_space.dim
    training_set_size = len(training_set)
    summary = '''Greedy basis generation:
   size of training set:   {training_set_size}
   error estimator used:   {use_estimator}
   extension method:       {extension_alg_name}
   prescribed basis size:  {max_extensions}
   actual basis size:      {real_rb_size}
   elapsed time:           {greedy_data[time]}
'''.format(**locals())

    return rd, rc, summary


def reduce_pod(d, reductor, snapshots_per_block, product_name, basis_size):
    from pymor.algorithms.pod import pod

    tic = time.time()

    training_set = d.parameter_space.sample_uniformly(snapshots_per_block)

    print('Solving on training set ...')
    snapshots = d.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        snapshots.append(d.solve(mu))

    if product_name == 'h1':
        pod_product = d.h1_0_semi_product
    elif product_name == 'trivial':
        pod_product = None
    else:
        assert False

    print('Performing POD ...')
    basis, singular_values = pod(snapshots, modes=basis_size, product=pod_product)

    print('Reducing ...')
    rd, rc, _ = reductor(d, basis)

    elapsed_time = time.time() - tic

    # generate summary
    real_rb_size = rd.solution_space.dim
    training_set_size = len(training_set)
    summary = '''POD basis generation:
   size of training set:   {training_set_size}
   inner product for POD:  {product_name}
   prescribed basis size:  {basis_size}
   actual basis size:      {real_rb_size}
   elapsed time:           {elapsed_time}
'''.format(**locals())

    return rd, rc, summary


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    thermalblock_demo(args)
