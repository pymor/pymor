#!/usr/bin/env python
# vim: set filetype=python:
'''Thermalblock demo.

Usage:
  thermalblock.py [-ehp] [--estimator-norm=NORM] [--extension-alg=ALG] [--grid=NI]
                  [--help] [--plot-solutions] [--test=COUNT] XBLOCKS YBLOCKS SNAPSHOTS RBSIZE


Arguments:
  XBLOCKS    Number of blocks in x direction.

  YBLOCKS    Number of blocks in y direction.

  SNAPSHOTS  Number of snapshots for basis generation per component.
             In total SNAPSHOTS^(XBLOCKS * YBLOCKS).

  RBSIZE     Size of the reduced basis


Options:
  -e, --with-estimator   Use error estimator.

  --estimator-norm=NORM  Norm (trivial, h1) in which to calculate the residual
                         [default: trivial].

  --extension-alg=ALG    Basis extension algorithm (trivial, gram_schmidt) to be used
                         [default: gram_schmidt].

  --grid=NI              Use grid with 2*NI*NI elements [default: 100].

  -h, --help             Show this message.

  --test=COUNT           Use COUNT snapshots for stochastic error estimation
                         [default: 10].
'''
from __future__ import absolute_import, division, print_function
import sys
from docopt import docopt
import time
from functools import partial
import math as m
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide import QtCore, QtGui

PARAM_STEPS = 10
PARAM_MIN = 0.01
PARAM_MAX = 1

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors.linear import reduce_stationary_affine_linear
from pymor.algorithms import greedy, trivial_basis_extension, gram_schmidt_basis_extension
from pymor.parameters.base import Parameter

# set log level
from pymor.core import getLogger
getLogger('pymor.algorithms').setLevel('DEBUG')
getLogger('pymor.discretizations').setLevel('DEBUG')

class ParamRuler(QtGui.QWidget):
    def __init__(self, parent, sim):
        super(ParamRuler, self).__init__(parent)
        self.sim = sim
        self.setMinimumSize(200, 100)
        box = QtGui.QGridLayout()
        self.spins = []
        for i in xrange(args['XBLOCKS']):
            for j in xrange(args['YBLOCKS']):
                spin = QtGui.QDoubleSpinBox()
                spin.setRange(PARAM_MIN, PARAM_MAX)
                spin.setSingleStep((PARAM_MAX - PARAM_MIN) / PARAM_STEPS)
                spin.setValue(PARAM_MIN)
                self.spins.append(spin)
                box.addWidget(spin, i, j)
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
        self.fig = Figure(figsize=(300, 300), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.ax = self.fig.add_subplot(111)
        # generate the canvas to display the plot
        self.canvas = FigureCanvas(self.fig)
        box.addWidget(self.canvas)
        box.addStretch()
        self.param_panel = ParamRuler(self, sim)
        box.addWidget(self.param_panel)
        self.setLayout(box)

    def solve_update(self):
        tic = time.time()
        self.param_panel.enable(False)
        args = self.sim.args
        shape = (args['XBLOCKS'], args['YBLOCKS'])
        mu = Parameter({'diffusion': np.array([s.value() for s in self.param_panel.spins]).reshape(shape)})
        print(mu)
        U = self.sim.solve(mu)
        print('Simtime {}'.format(time.time() - tic))
        tic = time.time()
        grid = self.sim.grid
        self.ax.clear()
        self.ax.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], U)
        self.fig.colorbar()
        self.canvas.draw()
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
        args['--test'] = int(args['--test'])
        args['--estimator-norm'] = args['--estimator-norm'].lower()
        assert args['--estimator-norm'] in {'trivial', 'h1'}
        args['--extension-alg'] = args['--extension-alg'].lower()
        assert args['--extension-alg'] in {'trivial', 'gram_schmidt'}
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
        extension_algorithm = (gram_schmidt_basis_extension if args['--extension-alg'] == 'gram_schmidt'
                       else trivial_basis_extension)
        greedy_data = greedy(self.discretization, reductor, self.discretization.parameter_space.sample_uniformly(args['SNAPSHOTS']),
                            use_estimator=args['--with-estimator'], error_norm=self.discretization.h1_norm,
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

    def solve(self, mu):
        return self.discretization.solve(mu)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    args = docopt(__doc__)
    win = RBGui(args)
    win.show()
    sys.exit(app.exec_())
