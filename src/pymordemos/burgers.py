#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Burgers demo.

Solves a two-dimensional Burgers-type equation. See pymor.analyticalproblems.burgers for more details.

Usage:
  burgers.py [-hp] [--grid=NI] [--grid-type=TYPE] [--initial-data=TYPE] [--lxf-lambda=VALUE] [--nt=COUNT]
             [--not-periodic] [--num-flux=FLUX] [--vx=XSPEED] [--vy=YSPEED] EXP


Arguments:
  EXP                    Exponent


Options:
  --grid=NI              Use grid with (2*NI)*NI elements [default: 60].

  --grid-type=TYPE       Type of grid to use (rect, tria) [default: rect].

  --initial-data=TYPE    Select the initial data (sin, bump) [default: sin]

  --lxf-lambda=VALUE     Parameter lambda in Lax-Friedrichs flux [default: 1].

  --nt=COUNT             Number of time steps [default: 100].

  --not-periodic         Solve with dirichlet boundary conditions on left
                         and bottom boundary.

  --num-flux=FLUX        Numerical flux to use (lax_friedrichs, engquist_osher)
                         [default: lax_friedrichs].

  -h, --help             Show this message.

  --vx=XSPEED            Speed in x-direction [default: 1].

  --vy=YSPEED            Speed in y-direction [default: 1].
"""

from __future__ import absolute_import, division, print_function

import sys
import math as m
import time
from functools import partial

from docopt import docopt

from pymor.analyticalproblems.burgers import Burgers2DProblem
from pymor.discretizers.advection import discretize_nonlinear_instationary_advection_fv
from pymor.domaindiscretizers.default import discretize_domain_default
from pymor.grids.rect import RectGrid
from pymor.grids.tria import TriaGrid


def burgers_demo(args):
    args['--grid'] = int(args['--grid'])
    args['--grid-type'] = args['--grid-type'].lower()
    assert args['--grid-type'] in ('rect', 'tria')
    args['--initial-data'] = args['--initial-data'].lower()
    assert args['--initial-data'] in ('sin', 'bump')
    args['--lxf-lambda'] = float(args['--lxf-lambda'])
    args['--nt'] = int(args['--nt'])
    args['--not-periodic'] = bool(args['--not-periodic'])
    args['--num-flux'] = args['--num-flux'].lower()
    assert args['--num-flux'] in ('lax_friedrichs', 'engquist_osher', 'simplified_engquist_osher')
    args['--vx'] = float(args['--vx'])
    args['--vy'] = float(args['--vy'])
    args['EXP'] = float(args['EXP'])

    print('Setup Problem ...')
    grid_type_map = {'rect': RectGrid, 'tria': TriaGrid}
    domain_discretizer = partial(discretize_domain_default, grid_type=grid_type_map[args['--grid-type']])
    problem = Burgers2DProblem(vx=args['--vx'], vy=args['--vy'], initial_data_type=args['--initial-data'],
                               parameter_range=(0, 1e42), torus=not args['--not-periodic'])

    print('Discretize ...')
    discretizer = discretize_nonlinear_instationary_advection_fv
    if args['--grid-type'] == 'rect':
        args['--grid'] *= 1. / m.sqrt(2)
    discretization, data = discretizer(problem, diameter=1. / args['--grid'],
                                       num_flux=args['--num-flux'], lxf_lambda=args['--lxf-lambda'],
                                       nt=args['--nt'], domain_discretizer=domain_discretizer)
    discretization.generate_sid()
    print(discretization.operator.grid)

    print('The parameter type is {}'.format(discretization.parameter_type))

    mu = args['EXP']
    # U = discretization.solve(0)
    print('Solving for exponent = {} ... '.format(mu))
    sys.stdout.flush()
    # pr = cProfile.Profile()
    # pr.enable()
    tic = time.time()
    U = discretization.solve(mu)
    # pr.disable()
    print('Solving took {}s'.format(time.time() - tic))
    # pr.dump_stats('bla')
    discretization.visualize(U)

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    burgers_demo(args)
