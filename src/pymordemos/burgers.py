#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Burgers demo.

Solves a two-dimensional Burgers-type equation. See pymor.analyticalproblems.burgers for more details.

Usage:
  burgers.py [-h] [--grid=NI] [--grid-type=TYPE] [--initial-data=TYPE] [--lxf-lambda=VALUE] [--nt=COUNT]
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
                         [default: engquist_osher].

  -h, --help             Show this message.

  --vx=XSPEED            Speed in x-direction [default: 1].

  --vy=YSPEED            Speed in y-direction [default: 1].
"""

import sys
import math as m
import time

from docopt import docopt

from pymor.analyticalproblems.burgers import burgers_problem_2d
from pymor.discretizers.fv import discretize_instationary_fv
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
    problem = burgers_problem_2d(vx=args['--vx'], vy=args['--vy'], initial_data_type=args['--initial-data'],
                                 parameter_range=(0, 1e42), torus=not args['--not-periodic'])

    print('Discretize ...')
    if args['--grid-type'] == 'rect':
        args['--grid'] *= 1. / m.sqrt(2)
    d, data = discretize_instationary_fv(
        problem,
        diameter=1. / args['--grid'],
        grid_type=RectGrid if args['--grid-type'] == 'rect' else TriaGrid,
        num_flux=args['--num-flux'],
        lxf_lambda=args['--lxf-lambda'],
        nt=args['--nt']
    )
    print(d.operator.grid)

    print('The parameter type is {}'.format(d.parameter_type))

    mu = args['EXP']
    print('Solving for exponent = {} ... '.format(mu))
    sys.stdout.flush()
    tic = time.time()
    U = d.solve(mu)
    print('Solving took {}s'.format(time.time() - tic))
    d.visualize(U)

if __name__ == '__main__':
    # parse arguments
    args = docopt(__doc__)
    # run demo
    burgers_demo(args)
