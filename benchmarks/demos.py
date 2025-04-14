# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from importlib import import_module

from typer import Typer
from typer.testing import CliRunner

runner = CliRunner()


class Demos:
    version = 1
    timeout = 600

    def setup(self):
        import sys
        sys._called_from_test = True
        try:
            from matplotlib import pyplot as plt
            plt.ion()
        except ImportError:
            pass
        try:
            import petsc4py

            # the default X handlers can interfere with process termination
            petsc4py.PETSc.Sys.popSignalHandler()
            petsc4py.PETSc.Sys.popErrorHandler()
        except ImportError:
            pass

    def teardown(self):
        from pymor.parallel.default import _cleanup
        _cleanup()

        from matplotlib import pyplot as plt
        plt.close('all')

    def run_demo(self, module, args):
        module = import_module(module)
        if hasattr(module, 'app'):
            app = module.app
        else:
            app = Typer()
            app.command()(module.main)
        args = [str(arg) for arg in args]
        runner.invoke(app, args, catch_exceptions=False)

    # thermalblock
    def time_thermalblock_small(self):
        self.run_demo('pymordemos.thermalblock', [2, 2, 2, 10])

    def time_thermalblock_highdim(self):
        self.run_demo('pymordemos.thermalblock', [2, 2, 2, 10, '--grid=300'])

    def time_thermalblock_manymu(self):
        self.run_demo('pymordemos.thermalblock', [2, 2, 16, 10, '--test=1'])

    def time_thermalblock_small_listva(self):
        self.run_demo('pymordemos.thermalblock', [2, 2, 2, 10, '--list-vector-array'])

    def time_thermalblock_highdim_listva(self):
        self.run_demo('pymordemos.thermalblock', [2, 2, 2, 10, '--grid=300', '--list-vector-array'])

    def time_thermalblock_manymu_listva(self):
        self.run_demo('pymordemos.thermalblock', [2, 2, 16, 10, '--test=1', '--list-vector-array'])

    # burgersei
    def time_burgersei_small(self):
        self.run_demo('pymordemos.burgers_ei', [1, 2, 3, 40, 3, 10, '--test=0', '--cache-region=disk'])

    def time_burgersei_highdim(self):
        self.run_demo('pymordemos.burgers_ei', [1, 2, 3, 40, 3, 10, '--test=0', '--cache-region=disk',
                                                '--grid=120', '--nt=200'])

    def time_burgersei_largecb(self):
        self.run_demo('pymordemos.burgers_ei', [1, 2, 10, 700, 3, 10, '--test=0', '--cache-region=disk'])

    # parabolic_mor
    def time_parabolic_mor(self):
        self.run_demo('pymordemos.parabolic_mor', ['pymor', 'greedy', 100, 50, 1, '--no-plot-err',
                                                   '--no-plot-error-sequence', '--no-pickle'])
