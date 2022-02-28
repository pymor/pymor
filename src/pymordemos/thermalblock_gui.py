#!/usr/bin/env python
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import time
import numpy as np
import OpenGL

from typer import Argument, Option, run

from pymor.core.config import is_windows_platform
from pymor.discretizers.builtin.gui.matplotlib import MatplotlibPatchWidget

OpenGL.ERROR_ON_COPY = True

from pymor.core.exceptions import QtMissing
try:
    from qtpy import QtWidgets
    from qtpy import QtCore
except ImportError as e:
    raise QtMissing() from e
from pymor.algorithms.greedy import rb_greedy
from pymor.analyticalproblems.thermalblock import thermal_block_problem
from pymor.discretizers.builtin import discretize_stationary_cg
from pymor.discretizers.builtin.gui.gl import ColorBarWidget, GLPatchWidget
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.tools.typer import Choices


PARAM_STEPS = 10
PARAM_MIN = 0.1
PARAM_MAX = 1


def main(
    xblocks: int = Argument(..., help='Number of blocks in x direction.'),
    yblocks: int = Argument(..., help='Number of blocks in y direction.'),
    snapshots: int = Argument(
        ...,
        help='Number of snapshots for basis generation per component. In total SNAPSHOTS^(XBLOCKS * YBLOCKS).'
    ),
    rbsize: int = Argument(..., help='Size of the reduced basis.'),

    grid: int = Option(60, help='Use grid with 2*NI*NI elements.'),
    product: Choices('euclidean h1') = Option(
        'h1',
        help='Product w.r.t. which to orthonormalize and calculate Riesz representatives.'
    ),
    testing: bool = Option(False, help='Load the gui and exit right away (for functional testing).'),
):
    """Thermalblock demo with GUI."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    win = RBGui(xblocks, yblocks, snapshots, rbsize, grid, product)
    win.show()
    if testing:
        QtCore.QTimer.singleShot(1000, app.quit)
    app.exec_()


class ParamRuler(QtWidgets.QWidget):
    def __init__(self, parent, sim):
        super().__init__(parent)
        self.sim = sim
        self.setMinimumSize(200, 100)
        box = QtWidgets.QGridLayout()
        self.spins = []
        for j in range(sim.xblocks):
            for i in range(sim.yblocks):
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
        tic = time.perf_counter()
        self.param_panel.enable(False)
        shape = (self.sim.yblocks, self.sim.xblocks)
        mu = {'diffusion': np.array([s.value() for s in self.param_panel.spins]).reshape(shape)}
        U = self.sim.solve(mu)
        print(f'Simtime {time.perf_counter()-tic}')
        tic = time.perf_counter()
        self.solution.set(U.to_numpy().ravel())
        self.param_panel.enable(True)
        print(f'Drawtime {time.perf_counter()-tic}')


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
    def __init__(self, *args):
        super().__init__()
        reduced = ReducedSim(*args)
        detailed = DetailedSim(*args)
        self.panel = AllPanel(self, reduced, detailed)
        self.setCentralWidget(self.panel)


# noinspection PyShadowingNames
class SimBase:
    def __init__(self, xblocks, yblocks, snapshots, rbsize, grid, product):
        self.snapshots, self.rbsize, self.product = snapshots, rbsize, product
        self.xblocks, self.yblocks = xblocks, yblocks
        self.first = True
        self.problem = thermal_block_problem(num_blocks=(xblocks, yblocks),
                                             parameter_range=(PARAM_MIN, PARAM_MAX))
        self.m, pack = discretize_stationary_cg(self.problem, diameter=1. / grid)
        self.grid = pack['grid']


# noinspection PyShadowingNames,PyShadowingNames
class ReducedSim(SimBase):

    def __init__(self, *args):
        super().__init__(*args)

    def _first(self):
        product = self.m.h1_0_semi_product if self.product == 'h1' else None
        reductor = CoerciveRBReductor(self.m, product=product)

        greedy_data = rb_greedy(self.m, reductor,
                                self.problem.parameter_space.sample_uniformly(self.snapshots),
                                use_error_estimator=True, error_norm=self.m.h1_0_semi_norm,
                                max_extensions=self.rbsize)
        self.rom, self.reductor = greedy_data['rom'], reductor
        self.first = False

    def solve(self, mu):
        if self.first:
            self._first()
        return self.reductor.reconstruct(self.rom.solve(mu))


# noinspection PyShadowingNames
class DetailedSim(SimBase):

    def __init__(self, *args):
        super().__init__(*args)
        self.m.disable_caching()

    def solve(self, mu):
        return self.m.solve(mu)


if __name__ == '__main__':
    run(main)
