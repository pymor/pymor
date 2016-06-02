#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Burgers with EI demo.

Model order reduction of a two-dimensional Burgers-type equation
(see pymor.analyticalproblems.burgers) using the reduced basis method
with empirical operator interpolation.

Usage:
  burgers_ei.py [options] EXP_MIN EXP_MAX EI_SNAPSHOTS EISIZE SNAPSHOTS RBSIZE


Arguments:
  EXP_MIN       Minimal exponent

  EXP_MAX       Maximal exponent

  EI_SNAPSHOTS  Number of snapshots for empirical interpolation.

  EISIZE        Number of interpolation DOFs.

  SNAPSHOTS     Number of snapshots for basis generation.

  RBSIZE        Size of the reduced basis


Options:
  --cache-region=REGION           Name of cache region to use for caching solution snapshots
                                  (NONE, MEMORY, DISK, PERSISTENT)
                                  [default: DISK]

  --grid=NI                       Use grid with (2*NI)*NI elements [default: 60].

  --grid-type=TYPE                Type of grid to use (rect, tria) [default: rect].

  --initial-data=TYPE             Select the initial data (sin, bump) [default: sin]

  --lxf-lambda=VALUE              Parameter lambda in Lax-Friedrichs flux [default: 1].

  --not-periodic                  Solve with dirichlet boundary conditions on left
                                  and bottom boundary.

  --nt=COUNT                      Number of time steps [default: 100].

  --num-flux=FLUX                 Numerical flux to use (lax_friedrichs, engquist_osher)
                                  [default: lax_friedrichs].

  -h, --help                      Show this message.

  -p, --plot-err                  Plot error.

  --plot-ei-err                   Plot empirical interpolation error.

  --plot-error-landscape          Calculate and show plot of reduction error vs. basis sizes.

  --plot-error-landscape-N=COUNT  Number of basis sizes to test [default: 10]

  --plot-error-landscape-M=COUNT  Number of collateral basis sizes to test [default: 10]

  --plot-solutions                Plot some example solutions.

  --test=COUNT                    Use COUNT snapshots for stochastic error estimation
                                  [default: 10].

  --vx=XSPEED                     Speed in x-direction [default: 1].

  --vy=YSPEED                     Speed in y-direction [default: 1].

  --ipython-engines=COUNT         If positive, the number of IPython cluster engines to use
                                  for parallel greedy search. If zero, no parallelization
                                  is performed. [default: 0]

  --ipython-profile=PROFILE       IPython profile to use for parallelization.
"""

import sys
import math as m
import time
from functools import partial

import numpy as np
from docopt import docopt

from pymor.algorithms.greedy import greedy
from pymor.algorithms.basisextension import pod_basis_extension
from pymor.algorithms.ei import interpolate_operators
from pymor.analyticalproblems.burgers import Burgers2DProblem
from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid
from pymor.parallel.default import new_parallel_pool
from pymor.reductors.basic import reduce_generic_rb, reduce_to_subbasis
from pymor.vectorarrays.numpy import NumpyVectorArray


def main(args):
    args = docopt(__doc__, args)
    args['--cache-region'] = args['--cache-region'].lower()
    args['--grid'] = int(args['--grid'])
    args['--grid-type'] = args['--grid-type'].lower()
    assert args['--grid-type'] in ('rect', 'tria')
    args['--initial-data'] = args['--initial-data'].lower()
    assert args['--initial-data'] in ('sin', 'bump')
    args['--lxf-lambda'] = float(args['--lxf-lambda'])
    args['--nt'] = int(args['--nt'])
    args['--not-periodic'] = bool(args['--not-periodic'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('lax_friedrichs', 'engquist_osher')
    args['--plot-error-landscape-N'] = int(args['--plot-error-landscape-N'])
    args['--plot-error-landscape-M'] = int(args['--plot-error-landscape-M'])
    args['--test'] = int(args['--test'])
    args['--vx'] = float(args['--vx'])
    args['--vy'] = float(args['--vy'])
    args['--ipython-engines'] = int(args['--ipython-engines'])
    args['EXP_MIN'] = int(args['EXP_MIN'])
    args['EXP_MAX'] = int(args['EXP_MAX'])
    args['EI_SNAPSHOTS'] = int(args['EI_SNAPSHOTS'])
    args['EISIZE'] = int(args['EISIZE'])
    args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
    args['RBSIZE'] = int(args['RBSIZE'])

    print('Setup Problem ...')
    grid_type_map = {'rect': RectGrid, 'tria': TriaGrid}
    domain_discretizer = partial(discretize_domain_default, grid_type=grid_type_map[args['--grid-type']])
    problem = Burgers2DProblem(vx=args['--vx'], vy=args['--vy'], initial_data_type=args['--initial-data'],
                               parameter_range=(args['EXP_MIN'], args['EXP_MAX']), torus=not args['--not-periodic'])

    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv
    if args['--grid-type'] == 'rect':
        args['--grid'] *= 1. / m.sqrt(2)
    discretization, _ = discretizer(problem, diameter=1. / args['--grid'],
                                    num_flux=args['--num-flux'], lxf_lambda=args['--lxf-lambda'],
                                    nt=args['--nt'], domain_discretizer=domain_discretizer)

    if args['--cache-region'] != 'none':
        discretization.enable_caching(args['--cache-region'])

    print(discretization.operator.grid)

    print('The parameter type is {}'.format(discretization.parameter_type))

    if args['--plot-solutions']:
        print('Showing some solutions')
        Us = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_uniformly(4):
            print('Solving for exponent = {} ... '.format(mu['exponent']))
            sys.stdout.flush()
            Us = Us + (discretization.solve(mu),)
            legend = legend + ('exponent: {}'.format(mu['exponent']),)
        discretization.visualize(Us, legend=legend, title='Detailed Solutions', block=True)

    pool = new_parallel_pool(ipython_num_engines=args['--ipython-engines'], ipython_profile=args['--ipython-profile'])
    ei_discretization, ei_data = interpolate_operators(discretization, ['operator'],
                                                       discretization.parameter_space.sample_uniformly(args['EI_SNAPSHOTS']),  # NOQA
                                                       error_norm=discretization.l2_norm,
                                                       max_interpolation_dofs=args['EISIZE'],
                                                       product=discretization.l2_product,
                                                       pool=pool)

    if args['--plot-ei-err']:
        print('Showing some EI errors')
        ERRs = tuple()
        legend = tuple()
        for mu in discretization.parameter_space.sample_randomly(2):
            print('Solving for exponent = \n{} ... '.format(mu['exponent']))
            sys.stdout.flush()
            U = discretization.solve(mu)
            U_EI = ei_discretization.solve(mu)
            ERR = U - U_EI
            ERRs = ERRs + (ERR,)
            legend = legend + ('exponent: {}'.format(mu['exponent']),)
            print('Error: {}'.format(np.max(discretization.l2_norm(ERR))))
        discretization.visualize(ERRs, legend=legend, title='EI Errors', separate_colorbars=True)

        print('Showing interpolation DOFs ...')
        U = np.zeros(U.dim)
        dofs = ei_discretization.operator.interpolation_dofs
        U[dofs] = np.arange(1, len(dofs) + 1)
        U[ei_discretization.operator.source_dofs] += int(len(dofs)/2)
        discretization.visualize(NumpyVectorArray(U), title='Interpolation DOFs')

    print('RB generation ...')

    def reductor(discretization, rb, extends=None):
        return reduce_generic_rb(ei_discretization, rb, extends=extends)

    extension_algorithm = partial(pod_basis_extension)

    greedy_data = greedy(discretization, reductor, discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                         use_estimator=False, error_norm=lambda U: np.max(discretization.l2_norm(U)),
                         extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'],
                         pool=pool)

    rb_discretization, reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']

    print('\nSearching for maximum error on random snapshots ...')

    tic = time.time()

    mus = discretization.parameter_space.sample_randomly(args['--test'])

    def error_analysis(N, M):
        print('N = {}, M = {}: '.format(N, M), end='')
        rd, rc, _ = reduce_to_subbasis(rb_discretization, N, reconstructor)
        rd = rd.with_(operator=rd.operator.projected_to_subbasis(dim_collateral=M))
        l2_err_max = -1
        mumax = None
        for mu in mus:
            print('.', end='')
            sys.stdout.flush()
            u = rd.solve(mu)
            URB = rc.reconstruct(u)
            U = discretization.solve(mu)
            l2_err = np.max(discretization.l2_norm(U - URB))
            l2_err = np.inf if not np.isfinite(l2_err) else l2_err
            if l2_err > l2_err_max:
                l2_err_max = l2_err
                mumax = mu
        print()
        return l2_err_max, mumax
    error_analysis = np.frompyfunc(error_analysis, 2, 2)

    real_rb_size = len(greedy_data['basis'])
    real_cb_size = len(ei_data['basis'])
    if args['--plot-error-landscape']:
        N_count = min(real_rb_size - 1, args['--plot-error-landscape-N'])
        M_count = min(real_cb_size - 1, args['--plot-error-landscape-M'])
        Ns = np.linspace(1, real_rb_size, N_count).astype(np.int)
        Ms = np.linspace(1, real_cb_size, M_count).astype(np.int)
    else:
        Ns = np.array([real_rb_size])
        Ms = np.array([real_cb_size])

    N_grid, M_grid = np.meshgrid(Ns, Ms)

    errs, err_mus = error_analysis(N_grid, M_grid)
    errs = errs.astype(np.float)

    l2_err_max = errs[-1, -1]
    mumax = err_mus[-1, -1]
    toc = time.time()
    t_est = toc - tic

    print('''
    *** RESULTS ***

    Problem:
       parameter range:                    ({args[EXP_MIN]}, {args[EXP_MAX]})
       h:                                  sqrt(2)/{args[--grid]}
       grid-type:                          {args[--grid-type]}
       initial-data:                       {args[--initial-data]}
       lxf-lambda:                         {args[--lxf-lambda]}
       nt:                                 {args[--nt]}
       not-periodic:                       {args[--not-periodic]}
       num-flux:                           {args[--num-flux]}
       (vx, vy):                           ({args[--vx]}, {args[--vy]})

    Greedy basis generation:
       number of ei-snapshots:             {args[EI_SNAPSHOTS]}
       prescribed collateral basis size:   {args[EISIZE]}
       actual collateral basis size:       {real_cb_size}
       number of snapshots:                {args[SNAPSHOTS]}
       prescribed basis size:              {args[RBSIZE]}
       actual basis size:                  {real_rb_size}
       elapsed time:                       {greedy_data[time]}

    Stochastic error estimation:
       number of samples:                  {args[--test]}
       maximal L2-error:                   {l2_err_max}  (mu = {mumax})
       elapsed time:                       {t_est}
    '''.format(**locals()))

    sys.stdout.flush()
    if args['--plot-error-landscape']:
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d             # NOQA
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # we have to rescale the errors since matplotlib does not support logarithmic scales on 3d plots
        # https://github.com/matplotlib/matplotlib/issues/209
        surf = ax.plot_surface(M_grid, N_grid, np.log(np.minimum(errs, 1)) / np.log(10),
                               rstride=1, cstride=1, cmap='jet')
        plt.show()
    if args['--plot-err']:
        U = discretization.solve(mumax)
        URB = reconstructor.reconstruct(rb_discretization.solve(mumax))
        discretization.visualize((U, URB, U - URB), legend=('Detailed Solution', 'Reduced Solution', 'Error'),
                                 title='Maximum Error Solution', separate_colorbars=True)

    return ei_data, greedy_data


if __name__ == '__main__':
    main(sys.argv[1:])
