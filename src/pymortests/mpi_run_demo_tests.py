# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

if __name__ == '__main__':
    from pymor.tools import mpi
    import pytest
    import os
    from pathlib import Path

    this_dir = Path(__file__).resolve().parent
    pymor_root_dir = (this_dir / '..' / '..').resolve()

    result_file_fn = pymor_root_dir / f'.mpirun_{mpi.rank}' / 'pytest.mpirun.success'
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
            # workaround for dolfin+dune incompat https://github.com/pymor/pymor/issues/1397
            import dune.gdt  # noqa
        except ImportError:
            pass
        try:
            import dolfin
            dolfin.plot = nop
        except ImportError:
            pass

        # completely disable FEniCS visualization on Gitlab CI
        import os
        if 'GITLAB_CI' in os.environ:
            from pymor.bindings.fenics import FenicsVisualizer

            FenicsVisualizer.visualize = nop

    mpi.call(monkey_plot)

    demo = str(pymor_root_dir / 'src' / 'pymortests' / 'demos.py')
    args = ['-svx', '-k', 'test_demo', demo]
    extra = os.environ.get('PYMOR_PYTEST_EXTRA', None)
    if extra:
        args.append(extra)
    success = pytest.main(args) == pytest.ExitCode.OK
    result_file_fn.parent.resolve().mkdir(parents=True, exist_ok=True)
    with open(result_file_fn, 'wt') as result_file:
        result_file.write(f'{success}')
