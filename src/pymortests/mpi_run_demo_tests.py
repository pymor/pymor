# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

if __name__ == '__main__':
    # ensure that we have no locale set, otherwise PETSc will fail when
    # matplotlib has been used before
    import os
    for k in ['LANG', 'LC_NUMERIC', 'LC_ALL']:
        if k in os.environ:
            del os.environ[k]

    from pymor.tools import mpi
    import runpy

    # ensure that FEniCS visualization does nothing on all MPI ranks
    def monkey_plot():
        def nop(*args, **kwargs):
            pass

        try:
            # for MPI runs we need to import qtgui before pyplot
            # otherwise, if both pyside and pyqt5 are installed we'll get an error later
            from Qt import QtGui
        except ImportError:
            pass
        try:
            from matplotlib import pyplot
            pyplot.show = nop
        except ImportError:
            pass

        try:
            import dolfin
            dolfin.plot = nop
        except ImportError:
            pass

    mpi.call(monkey_plot)

    runpy.run_module('pymortests.demos', init_globals=None, run_name='__main__', alter_sys=True)
