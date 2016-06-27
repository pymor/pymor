#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Thermalblock with GUI demo

Usage:
  thermalblock_gui.py [-h] [--estimator-norm=NORM] [--grid=NI] [--testing]
                  [--help] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: h1].

  --grid=NI              Use grid with 2*NI*NI elements [default: 60].

  --testing              load the gui and exit right away (for functional testing)

  -h, --help             Show this message.
"""

import sys
from docopt import docopt
import time
from functools import partial
import numpy as np
import OpenGL

OpenGL.ERROR_ON_COPY = True

from pymor.core.exceptions import PySideMissing
try:
    from PySide import QtGui
except ImportError as e:
    raise PySideMissing()
from pymor.algorithms.basisextension import gram_schmidt_basis_extension
from pymor.algorithms.greedy import greedy
from pymor.analyticalproblems.thermalblock import ThermalBlockProblem
from pymor.discretizers.elliptic import discretize_elliptic_cg
from pymor.gui.gl import ColorBarWidget, GLPatchWidget
from pymor.reductors.coercive import reduce_coercive_simple
from pymor import gui


PARAM_STEPS = 10
PARAM_MIN = 0.1
PARAM_MAX = 1


class ParamRuler(QtGui.QWidget):
    def __init__(self, parent, sim):
        super().__init__(parent)
        self.sim = sim
        self.setMinimumSize(200, 100)
        box = QtGui.QGridLayout()
        self.spins = []
        for j in range(args['YBLOCKS']):
            for i in range(args['XBLOCKS']):
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


# noinspection PyShadowingNames
class SimPanel(QtGui.QWidget):
    def __init__(self, parent, sim):
        super().__init__(parent)
        self.sim = sim
        box = QtGui.QHBoxLayout()
        self.solution = GLPatchWidget(self, self.sim.grid, vmin=0., vmax=0.8)
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
        super().__init__(parent)

        box = QtGui.QVBoxLayout()
        self.reduced_panel = SimPanel(self, reduced_sim)
        self.detailed_panel = SimPanel(self, detailed_sim)
        box.addWidget(self.reduced_panel)
        box.addWidget(self.detailed_panel)
        self.setLayout(box)


# noinspection PyShadowingNames
class RBGui(QtGui.QMainWindow):
    def __init__(self, args):
        super().__init__()
        args['XBLOCKS'] = int(args['XBLOCKS'])
        args['YBLOCKS'] = int(args['YBLOCKS'])
        args['--grid'] = int(args['--grid'])
        args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
        args['RBSIZE'] = int(args['RBSIZE'])
        args['--estimator-norm'] = args['--estimator-norm'].lower()
        assert args['--estimator-norm'] in {'trivial', 'h1'}
        reduced = ReducedSim(args)
        detailed = DetailedSim(args)
        self.panel = AllPanel(self, reduced, detailed)
        self.setCentralWidget(self.panel)


# noinspection PyShadowingNames
class SimBase(object):
    def __init__(self, args):
        self.args = args
        self.first = True
        self.problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']),
                                           parameter_range=(PARAM_MIN, PARAM_MAX))
        self.discretization, pack = discretize_elliptic_cg(self.problem, diameter=1. / args['--grid'])
        self.grid = pack['grid']


# noinspection PyShadowingNames,PyShadowingNames
class ReducedSim(SimBase):

    def __init__(self, args):
        super().__init__(args)

    def _first(self):
        args = self.args
        product = self.discretization.h1_0_semi_product if args['--estimator-norm'] == 'h1' else None
        reductor = partial(reduce_coercive_simple, product=product)
        extension_algorithm = partial(gram_schmidt_basis_extension, product=self.discretization.h1_0_semi_product)

        greedy_data = greedy(self.discretization, reductor,
                             self.discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                             use_estimator=True, error_norm=self.discretization.h1_0_semi_norm,
                             extension_algorithm=extension_algorithm, max_extensions=args['RBSIZE'])
        self.rb_discretization, self.reconstructor = greedy_data['reduced_discretization'], greedy_data['reconstructor']
        self.first = False

    def solve(self, mu):
        if self.first:
            self._first()
        return self.reconstructor.reconstruct(self.rb_discretization.solve(mu))


# noinspection PyShadowingNames
class DetailedSim(SimBase):

    def __init__(self, args):
        super().__init__(args)
        self.discretization.disable_caching()

    def solve(self, mu):
        return self.discretization.solve(mu)


if __name__ == '__main__':
    args = docopt(__doc__)
    testing = args['--testing']
    if not testing:
        app = QtGui.QApplication(sys.argv)
        win = RBGui(args)
        win.show()
        sys.exit(app.exec_())

    gui.qt._launch_qt_app(lambda _: RBGui(args), block=False)
