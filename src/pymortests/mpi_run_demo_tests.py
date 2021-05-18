# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

if __name__ == '__main__':
    from pymor.tools import mpi
    import pytest
    import os
    from pathlib import Path

    this_dir = Path(__file__).resolve().parent
    pymor_root_dir = (this_dir / '..' / '..').resolve()

    result_file_fn = pymor_root_dir / 'pytest.mpirun.success'
    try:
        os.unlink(result_file_fn)
    except FileNotFoundError:
        pass
    # ensure that FEniCS visualization does nothing on all MPI ranks
    def monkey_plot():
        def nop(*args, **kwargs):
            pass

        try:
            # for MPI runs we need to import qtgui before pyplot
            # otherwise, if both pyside and pyqt5 are installed we'll get an error later
            from qtpy import QtGui  # noqa F401
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

    demo = str(pymor_root_dir / 'src' / 'pymortests' / 'demos.py')
    success = pytest.main(['-svx', '-k', 'test_demo', demo]) == pytest.ExitCode.OK
    with open(result_file_fn, 'wt') as result_file:
        result_file.write(f'{success}')
