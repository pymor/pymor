#!/usr/bin/env python
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""Thermalblock with GUI demo

Usage:
  thermalblock_gui.py [-h] [--product=PROD] [--grid=NI] [--testing]
                  [--help] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.
  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  --grid=NI              Use grid with 2*NI*NI elements [default: 60].
  --product=PROD         Product (euclidean, h1) w.r.t. which to orthonormalize
                         and calculate Riesz representatives [default: h1].
  --testing              load the gui and exit right away (for functional testing)
  -h, --help             Show this message.
"""

import sys
from docopt import docopt
import time
import numpy as np
import OpenGL

from pymor.core.config import is_windows_platform
from pymor.discretizers.builtin.gui.matplotlib import MatplotlibPatchWidget

OpenGL.ERROR_ON_COPY = True

from pymor.core.exceptions import QtMissing
try:
    from Qt import QtWidgets
except ImportError as e:
    raise QtMissing()
from pymor.algorithms.greedy import rb_greedy
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.discretizers.builtin.gui.gl import ColorBarWidget, GLPatchWidget
from pymor.reductors.coercive import CoerciveRBReductor


PARAM_STEPS = 10
PARAM_MIN = 0.1
PARAM_MAX = 1


class ParamRuler(QtWidgets.QWidget):
    def __init__(self, parent, sim):
        super().__init__(parent)
        self.sim = sim
        self.setMinimumSize(200, 100)
        box = QtWidgets.QGridLayout()
        self.spins = []
        for j in range(args['YBLOCKS']):
            for i in range(args['XBLOCKS']):
                spin = QtWidgets.QDoubleSpinBox()
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
class SimPanel(QtWidgets.QWidget):
    def __init__(self, parent, sim):
        super().__init__(parent)
        self.sim = sim
        box = QtWidgets.QHBoxLayout()
        if is_windows_platform():
            self.solution = MatplotlibPatchWidget(self, self.sim.grid, vmin=0., vmax=0.8)
            box.addWidget(self.solution, 2)
        else:
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
        print(f'Simtime {time.time()-tic}')
        tic = time.time()
        self.solution.set(U.to_numpy().ravel())
        self.param_panel.enable(True)
        print(f'Drawtime {time.time()-tic}')


class AllPanel(QtWidgets.QWidget):
    def __init__(self, parent, reduced_sim, detailed_sim):
        super().__init__(parent)

        box = QtWidgets.QVBoxLayout()
        self.reduced_panel = SimPanel(self, reduced_sim)
        self.detailed_panel = SimPanel(self, detailed_sim)
        box.addWidget(self.reduced_panel)
        box.addWidget(self.detailed_panel)
        self.setLayout(box)


# noinspection PyShadowingNames
class RBGui(QtWidgets.QMainWindow):
    def __init__(self, args):
        super().__init__()
        args['XBLOCKS'] = int(args['XBLOCKS'])
        args['YBLOCKS'] = int(args['YBLOCKS'])
        args['--grid'] = int(args['--grid'])
        args['SNAPSHOTS'] = int(args['SNAPSHOTS'])
        args['RBSIZE'] = int(args['RBSIZE'])
        args['--product'] = args['--product'].lower()
        assert args['--product'] in {'trivial', 'h1'}
        reduced = ReducedSim(args)
        detailed = DetailedSim(args)
        self.panel = AllPanel(self, reduced, detailed)
        self.setCentralWidget(self.panel)


# noinspection PyShadowingNames
class SimBase:
    def __init__(self, args):
        self.args = args
        self.first = True
        self.problem = thermal_block_problem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']),
                                             parameter_range=(PARAM_MIN, PARAM_MAX))
        self.m, pack = discretize_stationary_cg(self.problem, diameter=1. / args['--grid'])
        self.grid = pack['grid']


# noinspection PyShadowingNames,PyShadowingNames
class ReducedSim(SimBase):

    def __init__(self, args):
        super().__init__(args)

    def _first(self):
        args = self.args
        product = self.m.h1_0_semi_product if args['--product'] == 'h1' else None
        reductor = CoerciveRBReductor(self.m, product=product)

        greedy_data = rb_greedy(self.m, reductor,
                                self.problem.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                                use_estimator=True, error_norm=self.m.h1_0_semi_norm,
                                max_extensions=args['RBSIZE'])
        self.rom, self.reductor = greedy_data['rom'], reductor
        self.first = False

    def solve(self, mu):
        if self.first:
            self._first()
        return self.reductor.reconstruct(self.rom.solve(mu))


# noinspection PyShadowingNames
class DetailedSim(SimBase):

    def __init__(self, args):
        super().__init__(args)
        self.m.disable_caching()

    def solve(self, mu):
        return self.m.solve(mu)


if __name__ == '__main__':
    args = docopt(__doc__)
    testing = args['--testing']
    if not testing:
        app = QtWidgets.QApplication(sys.argv)
        win = RBGui(args)
        win.show()
        sys.exit(app.exec_())

    from pymor.discretizers.builtin.gui import qt
    qt._launch_qt_app(lambda : RBGui(args), block=False)
