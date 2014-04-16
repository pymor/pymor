#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''Thermalblock with GUI demo

Usage:
  thermalblock_gui.py [-h] [--estimator-norm=NORM] [--grid=NI]
                  [--help] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: trivial].

  --grid=NI              Use grid with 2*NI*NI elements [default: 60].

  -h, --help             Show this message.
'''

from __future__ import absolute_import, division, print_function
import sys
from docopt import docopt
import time
from functools import partial
import math as m
import numpy as np
from PySide import QtGui
import OpenGL
OpenGL.ERROR_ON_COPY = True

import pymor.core as core
core.logger.MAX_HIERACHY_LEVEL = 2
from pymor.algorithms import greedy, gram_schmidt_basis_extension
from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.gui.glumpy import ColorBarWidget, GlumpyPatchWidget
from pymor.reductors.linear import reduce_stationary_affine_linear

core.getLogger('pymor.algorithms').setLevel('DEBUG')
core.getLogger('pymor.discretizations').setLevel('DEBUG')

PARAM_STEPS = 10
PARAM_MIN = 0.1
PARAM_MAX = 1


class ParamRuler(QtGui.QWidget):
    def __init__(self, parent, sim):
        super(ParamRuler, self).__init__(parent)
        self.sim = sim
        self.setMinimumSize(200, 100)
        box = QtGui.QGridLayout()
        self.spins = []
        for j in xrange(args['YBLOCKS']):
            for i in xrange(args['XBLOCKS']):
                spin = QtGui.QDoubleSpinBox()
                spin.setRange(PARAM_MIN, PARAM_MAX)
                spin.setSingleStep((PARAM_MAX - PARAM_MIN) / PARAM_STEPS)
                spin.setValue(PARAM_MIN)
                self.spins.append(spin)
                box.addWidget(spin, j, i)
                spin.valueChanged.connect(parent.solve_update)
        self.setLayout(box)

    def enable(self, enable=True):
        for spin in self.spins:
            spin.isEnabled = enable


class SimPanel(QtGui.QWidget):
    def __init__(self, parent, sim):
        super(SimPanel, self).__init__(parent)
        self.sim = sim
        box = QtGui.QHBoxLayout()
        self.solution = GlumpyPatchWidget(self, self.sim.grid, vmin=0., vmax=0.8)
        self.bar = ColorBarWidget(self, vmin=0., vmax=0.8)
        box.addWidget(self.solution, 2)
        box.addWidget(self.bar, 2)
        self.param_panel = ParamRuler(self, sim)
        box.addWidget(self.param_panel)
        self.setLayout(box)

    def solve_update(self):
        tic = time.time()
        self.param_panel.enable(False)
        args = self.sim.args
        shape = (args['YBLOCKS'], args['XBLOCKS'])
        mu = {'diffusion': np.array([s.value() for s in self.param_panel.spins]).reshape(shape)}
        U = self.sim.solve(mu)
        print('Simtime {}'.format(time.time() - tic))
        tic = time.time()
        self.solution.set(U.data.ravel())
        self.param_panel.enable(True)
        print('Drawtime {}'.format(time.time() - tic))


class AllPanel(QtGui.QWidget):
    def __init__(self, parent, reduced_sim, detailed_sim):
        super(AllPanel, self).__init__(parent)

        box = QtGui.QVBoxLayout()
        box.addWidget(SimPanel(self, reduced_sim))
        box.addWidget(SimPanel(self, detailed_sim))
        self.setLayout(box)


class RBGui(QtGui.QMainWindow):
    def __init__(self, args):
        super(RBGui, self).__init__()
        args['XBLOCKS'] = int(args['XBLOCKS'])
        args['YBLOCKS'] = int(args['YBLOCKS'])
        args['--grid'] = int(args['--grid'])
        args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
        args['RBSIZE'] = int(args['RBSIZE'])
        args['--estimator-norm'] = args['--estimator-norm'].lower()
        assert args['--estimator-norm'] in {'trivial', 'h1'}
        reduced = ReducedSim(args)
        detailed = DetailedSim(args)
        panel = AllPanel(self, reduced, detailed)
        self.setCentralWidget(panel)


class SimBase(object):
    def __init__(self, args):
        self.args = args
        self.first = True
        self.problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']),
                                           parameter_range=(PARAM_MIN, PARAM_MAX))
        self.discretization, pack = discretize_elliptic_cg(self.problem, diameter=m.sqrt(2) / args['--grid'])
        self.grid = pack['grid']


class ReducedSim(SimBase):

    def __init__(self, args):
        super(ReducedSim, self).__init__(args)

    def _first(self):
        args = self.args
        error_product = self.discretization.h1_product if args['--estimator-norm'] == 'h1' else None
        reductor = partial(reduce_stationary_affine_linear, error_product=error_product)
        extension_algorithm = partial(gram_schmidt_basis_extension, product=self.discretization.h1_product)

        greedy_data = greedy(self.discretization, reductor,
                             self.discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                             use_estimator=True, error_norm=self.discretization.h1_norm,
                             extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
        self.rb_discretization, self.reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']
        self.first = False

    def solve(self, mu):
        if self.first:
            self._first()
        return self.reconstructor.reconstruct(self.rb_discretization.solve(mu))


class DetailedSim(SimBase):

    def __init__(self, args):
        super(DetailedSim, self).__init__(args)
        self.discretization.disable_caching()

    def solve(self, mu):
        return self.discretization.solve(mu)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    args = docopt(__doc__)
    win = RBGui(args)
    win.show()
    sys.exit(app.exec_())
