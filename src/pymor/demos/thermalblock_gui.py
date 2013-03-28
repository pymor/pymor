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

import sys
from docopt import docopt
import math as m
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide import QtCore, QtGui

PARAM_STEPS = 10

from pymor.analyticalproblems import ThermalBlockProblem
from pymor.discretizers import discretize_elliptic_cg
from pymor.reductors.linear import reduce_stationary_affine_linear
from pymor.algorithms import greedy, trivial_basis_extension, gram_schmidt_basis_extension

# set log level
from pymor.core import getLogger
getLogger('pymor.algorithms').setLevel('INFO')
getLogger('pymor.discretizations').setLevel('INFO')

class ParamRuler(QtGui.QWidget):
    def __init__(self, parent):
        super(ParamRuler, self).__init__(parent)
        self.setMinimumSize(200, 100)
        box = QtGui.QVBoxLayout()
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(PARAM_STEPS)
        self.slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        # self.slider.setTickInterval()
        self.slider.setSingleStep(1)
        box.addWidget(self.slider)
        self.setLayout(box)
        self.slider.valueChanged.connect(parent.solve_update)

class SimPanel(QtGui.QWidget):
    def __init__(self, parent, sim):
        super(SimPanel, self).__init__(parent)
        self.sim = sim
        box = QtGui.QHBoxLayout()
        self.fig = Figure(figsize=(300, 300), dpi=72, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([0, 1])
        # generate the canvas to display the plot
        self.canvas = FigureCanvas(self.fig)
        box.addWidget(self.canvas)
        box.addStretch()
        self.param_panel = ParamRuler(self)
        box.addWidget(self.param_panel)
        self.setLayout(box)

    def solve_update(self):
        U = self.sim.solve(self.sim.params[self.param_panel.slider.value()])
        grid = self.sim.grid
        self.ax.tripcolor(grid.centers(2)[:, 0], grid.centers(2)[:, 1], U)
        # self.ax.colorbar()
        self.canvas.draw()


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
        # reduced = ReducedSim(args)

        detailed = DetailedSim(args)
        reduced = DetailedSim(args)
        panel = AllPanel(self, reduced, detailed)
        self.setCentralWidget(panel)

class ReducedSim(object):
    
    def __init__(self, args):
        self.args = args

    def solve(self, mu):
        pass

class DetailedSim(object):

    def __init__(self, args):
        self.args = args
        self.problem = ThermalBlockProblem(num_blocks=(args['XBLOCKS'], args['YBLOCKS']))
        self.discretization, pack = discretize_elliptic_cg(self.problem, diameter=m.sqrt(2) / args['--grid'])
        self.grid = pack['grid']
        self.params = list(self.problem.parameter_space.sample_uniformly(PARAM_STEPS))


    def solve(self, mu):
        return self.discretization.solve(mu)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    args = docopt(__doc__)
    win = RBGui(args)
    win.show()
    sys.exit(app.exec_())
