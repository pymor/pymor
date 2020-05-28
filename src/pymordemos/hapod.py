#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""HAPOD demo.

Demonstrates compression of snapshot data with the HAPOD algorithm from [HLR18].

Usage:
  hapod.py [options] TOL DIST INC


Arguments:
  TOL                    Prescribed mean l2 approximation error.
  DIST                   Number of slices for distributed HAPOD.
  INC                    Number of steps for incremental HAPOD.

Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].
  -h, --help             Show this message.
  --nt=COUNT             Number of time steps [default: 100].
  --omega=OMEGA          Parameter omega from HAPOD algorithm [default: 0.9].
  --procs=PROCS          Number of processes to use for parallelization [default: 0].
  --snap=SNAP            Number of snapshot trajectories to compute [default: 20].
  --threads=THREADS      Number of threads to use for parallelization [default: 0].
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from time import time

from docopt import docopt
import numpy as np

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.builtin import discretize_instationary_fv, RectGrid
from pymor.algorithms.hapod import dist_vectorarray_hapod, inc_vectorarray_hapod
from pymor.algorithms.pod import pod
from pymor.tools.table import format_table


def hapod_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--nt'] = int(args['--nt'])
    args['--omega'] = float(args['--omega'])
    args['--procs'] = int(args['--procs'])
    args['--snap'] = int(args['--snap'])
    args['--threads'] = int(args['--threads'])
    args['TOL'] = float(args['TOL'])
    args['DIST'] = int(args['DIST'])
    args['INC'] = int(args['INC'])
    assert args['--procs'] == 0 or args['--threads'] == 0

    tol = args['TOL']
    omega = args['--omega']
    executor = ProcessPoolExecutor(args['--procs']) if args['--procs'] > 0 else \
        ThreadPoolExecutor(args['--threads']) if args['--threads'] > 0 else \
        None

    p = burgers_problem_2d()
    m, data = discretize_instationary_fv(p, grid_type=RectGrid, diameter=np.sqrt(2)/args['--grid'], nt=args['--nt'])

    U = m.solution_space.empty()
    for mu in p.parameter_space.sample_randomly(args['--snap']):
        U.append(m.solve(mu))

    tic = time()
    pod_modes = pod(U, l2_err=tol * np.sqrt(len(U)), product=m.l2_product)[0]
    pod_time = time() - tic

    tic = time()
    dist_modes = dist_vectorarray_hapod(args['DIST'], U, tol, omega, product=m.l2_product, executor=executor)[0]
    dist_time = time() - tic

    tic = time()
    inc_modes = inc_vectorarray_hapod(args['INC'], U, tol, omega, product=m.l2_product)[0]
    inc_time = time() - tic

    print(f'Snapshot matrix: {U.dim} x {len(U)}')
    print(format_table([
        ['Method', 'Error', 'Modes', 'Time'],
        ['POD', np.linalg.norm(m.l2_norm(U-pod_modes.lincomb(m.l2_product.apply2(U, pod_modes)))/np.sqrt(len(U))),
         len(pod_modes), pod_time],
        ['DIST HAPOD', np.linalg.norm(m.l2_norm(U-dist_modes.lincomb(m.l2_product.apply2(U, dist_modes)))/np.sqrt(len(U))),
         len(dist_modes), dist_time],
        ['INC HAPOD', np.linalg.norm(m.l2_norm(U-inc_modes.lincomb(m.l2_product.apply2(U, inc_modes)))/np.sqrt(len(U))),
         len(inc_modes), inc_time]]
    ))


if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    hapod_demo(args)
