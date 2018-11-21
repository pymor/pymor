#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Thermalblock demo.

Usage:
  thermalblock.py [options] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE
  thermalblock.py -h | --help


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  naive:           ignored

             greedy/pod:      Number of training_set parameters per block
                              (in total SNAPSHOTS^(XBLOCKS * YBLOCKS)
                              parameters).

             adaptive_greedy: size of validation set.

  RBSIZE     Size of the reduced basis


Options:
  --adaptive-greedy-rho=RHO       See pymor.algorithms.adaptivegreedy [default: 1.1].

  --adaptive-greedy-gamma=GAMMA   See pymor.algorithms.adaptivegreedy [default: 0.2].

  --adaptive-greedy-theta=THETA   See pymor.algorithms.adaptivegreedy [default: 0.]

  --alg=ALG                       The model reduction algorithm to use
                                  (naive, greedy, adaptive_greedy, pod) [default: greedy].

  --cache-region=REGION           Name of cache region to use for caching solution snapshots
                                  (none, memory, disk, persistent) [default: none].

  --extension-alg=ALG             Basis extension algorithm (trivial, gram_schmidt)
                                  to be used [default: gram_schmidt].

  --fenics                        Use FEniCS discretization.

  --grid=NI                       Use grid with 4*NI*NI elements [default: 100].

  -h, --help                      Show this message.

  --ipython-engines=COUNT         If positive, the number of IPython cluster engines to use for
                                  parallel greedy search. If zero, no parallelization is performed.
                                  [default: 0]

  --ipython-profile=PROFILE       IPython profile to use for parallelization.

  --list-vector-array             Solve using ListVectorArray[NumpyVector] instead of NumpyVectorArray.

  --order=ORDER                   Polynomial order of the Lagrange finite elements to use in FEniCS
                                  discretization [default: 1].

  --pickle=PREFIX                 Pickle reduced discretizaion, as well as reductor and high-dimensional
                                  discretization to files with this prefix.

  --plot-err                      Plot error.

  --plot-solutions                Plot some example solutions.

  --plot-error-sequence           Plot reduction error vs. basis size.

  --product=PROD                  Product (euclidean, h1) w.r.t. which to orthonormalize
                                  and calculate Riesz representatives [default: h1].

  --reductor=RED                  Reductor (error estimator) to choose (traditional, residual_basis)
                                  [default: residual_basis]

  --test=COUNT                    Use COUNT snapshots for stochastic error estimation
                                  [default: 10].

  --greedy-without-estimator      Do not use error estimator for basis generation.
"""

import sys
import time

from docopt import docopt

from pymor.algorithms.error import reduction_error_analysis
from pymor.core.pickle import dump
from pymor.parallel.default import new_parallel_pool


def main(args):

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
        Us = ()
        legend = ()
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
    product = d.h1_0_semi_product if args['--product'] == 'h1' else None

    if args['--reductor'] == 'residual_basis':
        from pymor.reductors.coercive import CoerciveRBReductor
        reductor = CoerciveRBReductor(d, product=product, coercivity_estimator=coercivity_estimator)
    elif args['--reductor'] == 'traditional':
        from pymor.reductors.coercive import SimpleCoerciveRBReductor
        reductor = SimpleCoerciveRBReductor(d, product=product, coercivity_estimator=coercivity_estimator)
    else:
        assert False  # this should never happen

    if args['--alg'] == 'naive':
        rd, red_summary = reduce_naive(d=d, reductor=reductor, basis_size=args['RBSIZE'])
    elif args['--alg'] == 'greedy':
        parallel = not (args['--fenics'] and args['--greedy-without-estimator'])  # cannot pickle FEniCS discretization
        rd, red_summary = reduce_greedy(d=d, reductor=reductor, snapshots_per_block=args['SNAPSHOTS'],
                                        extension_alg_name=args['--extension-alg'],
                                        max_extensions=args['RBSIZE'],
                                        use_estimator=not args['--greedy-without-estimator'],
                                        pool=pool if parallel else None)
    elif args['--alg'] == 'adaptive_greedy':
        parallel = not (args['--fenics'] and args['--greedy-without-estimator'])  # cannot pickle FEniCS discretization
        rd, red_summary = reduce_adaptive_greedy(d=d, reductor=reductor, validation_mus=args['SNAPSHOTS'],
                                                 extension_alg_name=args['--extension-alg'],
                                                 max_extensions=args['RBSIZE'],
                                                 use_estimator=not args['--greedy-without-estimator'],
                                                 rho=args['--adaptive-greedy-rho'],
                                                 gamma=args['--adaptive-greedy-gamma'],
                                                 theta=args['--adaptive-greedy-theta'],
                                                 pool=pool if parallel else None)
    elif args['--alg'] == 'pod':
        rd, red_summary = reduce_pod(d=d, reductor=reductor, snapshots_per_block=args['SNAPSHOTS'],
                                     basis_size=args['RBSIZE'])
    else:
        assert False  # this should never happen

    if args['--pickle']:
        print('\nWriting reduced discretization to file {} ...'.format(args['--pickle'] + '_reduced'))
        with open(args['--pickle'] + '_reduced', 'wb') as f:
            dump(rd, f)
        if not args['--fenics']:  # FEniCS data structures do not support serialization
            print('Writing detailed discretization and reductor to file {} ...'
                  .format(args['--pickle'] + '_detailed'))
            with open(args['--pickle'] + '_detailed', 'wb') as f:
                dump((d, reductor), f)

    print('\nSearching for maximum error on random snapshots ...')

    results = reduction_error_analysis(rd,
                                       d=d,
                                       reductor=reductor,
                                       estimator=True,
                                       error_norms=(d.h1_0_semi_norm, d.l2_norm),
                                       condition=True,
                                       test_mus=args['--test'],
                                       basis_sizes=0 if args['--plot-error-sequence'] else 1,
                                       plot=args['--plot-error-sequence'],
                                       pool=None if args['--fenics'] else pool,  # cannot pickle FEniCS discretization
                                       random_seed=999)

    print('\n*** RESULTS ***\n')
    print(d_summary)
    print(red_summary)
    print(results['summary'])
    sys.stdout.flush()

    if args['--plot-error-sequence']:
        import matplotlib.pyplot
        matplotlib.pyplot.show(results['figure'])
    if args['--plot-err']:
        mumax = results['max_error_mus'][0, -1]
        U = d.solve(mumax)
        URB = reductor.reconstruct(rd.solve(mumax))
        d.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                    title='Maximum Error Solution', separate_colorbars=True, block=True)

    return results


def parse_arguments(args):
    args = docopt(__doc__, args)
    args['XBLOCKS'] = int(args['XBLOCKS'])
    args['YBLOCKS'] = int(args['YBLOCKS'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])

    args['--adaptive-greedy-rho'] = float(args['--adaptive-greedy-rho'])
    args['--adaptive-greedy-gamma'] = float(args['--adaptive-greedy-gamma'])
    args['--adaptive-greedy-theta'] = float(args['--adaptive-greedy-theta'])
    args['--alg'] = args['--alg'].lower()
    args['--cache-region'] = args['--cache-region'].lower()
    args['--extension-alg'] = args['--extension-alg'].lower()
    args['--grid'] = int(args['--grid'])
    args['--ipython-engines'] = int(args['--ipython-engines'])
    args['--order'] = int(args['--order'])
    args['--product'] = args['--product'].lower()
    args['--reductor'] = args['--reductor'].lower()
    args['--test'] = int(args['--test'])

    assert args['--alg'] in {'naive', 'greedy', 'adaptive_greedy', 'pod'}
    assert args['--cache-region'] in {'none', 'memory', 'disk', 'persistent'}
    assert args['--extension-alg'] in {'trivial', 'gram_schmidt', 'h1_gram_schmidt'}
    assert args['--product'] in {'euclidean', 'h1'}
    assert args['--reductor'] in {'traditional', 'residual_basis'}

    if args['--fenics']:
        if args['--cache-region'] != 'none':
            raise ValueError('Caching of high-dimensional solutions is not supported for FEniCS discretization.')
    else:
        if args['--order'] != 1:
            raise ValueError('Higher-order finite elements only supported for FEniCS discretization.')

    return args


def discretize_pymor(xblocks, yblocks, grid_num_intervals, use_list_vector_array):
    from pymor.analyticalproblems.thermalblock import thermal_block_problem
    from pymor.discretizers.cg import discretize_stationary_cg
    from pymor.playground.discretizers.numpylistvectorarray import convert_to_numpy_list_vector_array

    print('Discretize ...')
    # setup analytical problem
    problem = thermal_block_problem(num_blocks=(xblocks, yblocks))

    # discretize using continuous finite elements
    d, _ = discretize_stationary_cg(problem, diameter=1. / grid_num_intervals)

    if use_list_vector_array:
        d = convert_to_numpy_list_vector_array(d)

    summary = '''pyMOR discretization:
   number of blocks: {xblocks}x{yblocks}
   grid intervals:   {grid_num_intervals}
   ListVectorArray:  {use_list_vector_array}
'''.format(**locals())

    return d, summary


def discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):
    from pymor.tools import mpi

    if mpi.parallel:
        from pymor.discretizations.mpi import mpi_wrap_discretization
        d = mpi_wrap_discretization(lambda: _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order),
                                    use_with=True, pickle_local_spaces=False)
    else:
        d = _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order)

    summary = '''FEniCS discretization:
   number of blocks:      {xblocks}x{yblocks}
   grid intervals:        {grid_num_intervals}
   finite element order:  {element_order}
'''.format(**locals())

    return d, summary


def _discretize_fenics(xblocks, yblocks, grid_num_intervals, element_order):

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
    from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator, FenicsVisualizer

    # generic pyMOR classes
    from pymor.discretizations.basic import StationaryDiscretization
    from pymor.operators.constructions import LincombOperator, VectorOperator
    from pymor.parameters.functionals import ProjectionParameterFunctional
    from pymor.parameters.spaces import CubicParameterSpace

    # define parameter functionals (same as in pymor.analyticalproblems.thermalblock)
    def parameter_functional_factory(x, y):
        return ProjectionParameterFunctional(component_name='diffusion',
                                             component_shape=(yblocks, xblocks),
                                             coordinates=(yblocks - y - 1, x),
                                             name='diffusion_{}_{}'.format(x, y))
    parameter_functionals = tuple(parameter_functional_factory(x, y)
                                  for x in range(xblocks) for y in range(yblocks))

    # wrap operators
    ops = [FenicsMatrixOperator(mat0, V, V)] + [FenicsMatrixOperator(m, V, V) for m in mats]
    op = LincombOperator(ops, (1.,) + parameter_functionals)
    rhs = VectorOperator(FenicsVectorSpace(V).make_array([F]))
    h1_product = FenicsMatrixOperator(h1_mat, V, V, name='h1_0_semi')
    l2_product = FenicsMatrixOperator(l2_mat, V, V, name='l2')

    # build discretization
    visualizer = FenicsVisualizer(FenicsVectorSpace(V))
    parameter_space = CubicParameterSpace(op.parameter_type, 0.1, 1.)
    d = StationaryDiscretization(op, rhs, products={'h1_0_semi': h1_product,
                                                    'l2': l2_product},
                                 parameter_space=parameter_space,
                                 visualizer=visualizer)

    return d


def reduce_naive(d, reductor, basis_size):

    tic = time.time()

    training_set = d.parameter_space.sample_randomly(basis_size)

    for mu in training_set:
        reductor.extend_basis(d.solve(mu), 'trivial')

    rd = reductor.reduce()

    elapsed_time = time.time() - tic

    summary = '''Naive basis generation:
   basis size set: {basis_size}
   elapsed time:   {elapsed_time}
'''.format(**locals())

    return rd, summary


def reduce_greedy(d, reductor, snapshots_per_block,
                  extension_alg_name, max_extensions, use_estimator, pool):

    from pymor.algorithms.greedy import greedy

    # run greedy
    training_set = d.parameter_space.sample_uniformly(snapshots_per_block)
    greedy_data = greedy(d, reductor, training_set,
                         use_estimator=use_estimator, error_norm=d.h1_0_semi_norm,
                         extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                         pool=pool)
    rd = greedy_data['rd']

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

    return rd, summary


def reduce_adaptive_greedy(d, reductor, validation_mus,
                           extension_alg_name, max_extensions, use_estimator,
                           rho, gamma, theta, pool):

    from pymor.algorithms.adaptivegreedy import adaptive_greedy

    # run greedy
    greedy_data = adaptive_greedy(d, reductor, validation_mus=-validation_mus,
                                  use_estimator=use_estimator, error_norm=d.h1_0_semi_norm,
                                  extension_params={'method': extension_alg_name}, max_extensions=max_extensions,
                                  rho=rho, gamma=gamma, theta=theta, pool=pool)
    rd = greedy_data['rd']

    # generate summary
    real_rb_size = rd.solution_space.dim
    # the validation set consists of `validation_mus` random parameters plus the centers of the adaptive sample set cells
    validation_mus += 1
    summary = '''Adaptive greedy basis generation:
   initial size of validation set:  {validation_mus}
   error estimator used:            {use_estimator}
   extension method:                {extension_alg_name}
   prescribed basis size:           {max_extensions}
   actual basis size:               {real_rb_size}
   elapsed time:                    {greedy_data[time]}
'''.format(**locals())

    return rd, summary


def reduce_pod(d, reductor, snapshots_per_block, basis_size):
    from pymor.algorithms.pod import pod

    tic = time.time()

    training_set = d.parameter_space.sample_uniformly(snapshots_per_block)

    print('Solving on training set ...')
    snapshots = d.operator.source.empty(reserve=len(training_set))
    for mu in training_set:
        snapshots.append(d.solve(mu))

    print('Performing POD ...')
    basis, singular_values = pod(snapshots, modes=basis_size, product=reductor.product)

    print('Reducing ...')
    reductor.extend_basis(basis, 'trivial')
    rd = reductor.reduce()

    elapsed_time = time.time() - tic

    # generate summary
    real_rb_size = rd.solution_space.dim
    training_set_size = len(training_set)
    summary = '''POD basis generation:
   size of training set:   {training_set_size}
   prescribed basis size:  {basis_size}
   actual basis size:      {real_rb_size}
   elapsed time:           {elapsed_time}
'''.format(**locals())

    return rd, summary


if __name__ == '__main__':
    main(sys.argv[1:])
