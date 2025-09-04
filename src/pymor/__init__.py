# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

__version__ = '2025.1.0'
__asv_api_gen__ = 1  # used by asv benchmarks to determine to adapt test code in case of api breakage

import os
import platform
import sys


def _init_mpi():
    """Provides a way to manually set the thread init mode for MPI if necessary.

    Needs to happen as early as possible, otherwise mpi4py might auto-init somewhere else.
    """
    try:
        import mpi4py
    except ImportError:
        return

    finalize = os.environ.get('PYMOR_MPI_FINALIZE', mpi4py.rc.finalize if mpi4py.rc.finalize is not None else True)
    mpi4py.rc(initialize=False, finalize=finalize)
    try:
        from mpi4py import MPI
    except RuntimeError:
        return
    if not MPI.Is_initialized():
        required_level = int(os.environ.get('PYMOR_MPI_INIT_THREAD', MPI.THREAD_MULTIPLE))
        supported_lvl = MPI.Init_thread(required_level)
        if supported_lvl < required_level:
            print(f'MPI does support threading level {required_level},\
                   running with {supported_lvl} instead', flush=True)

    try:
        # this solves sporadic mpi calls happening after finalize
        import petsc4py
        petsc4py.init()
    except ImportError:
        return

_init_mpi()

from pymor.core.config import config, is_jupyter
from pymor.core.defaults import defaults, load_defaults_from_file

if 'PYMOR_DEFAULTS' in os.environ:
    filename = os.environ['PYMOR_DEFAULTS']
    if filename in ('', 'NONE'):
        print('Not loading any pyMOR defaults from config file')
    else:
        for fn in filename.split(':'):
            if not os.path.exists(fn):
                raise OSError('Cannot load pyMOR defaults from file ' + fn)
            print('Loading pyMOR defaults from file ' + fn + ' (set by PYMOR_DEFAULTS)')
            load_defaults_from_file(fn)
else:
    filename = os.path.join(os.getcwd(), 'pymor_defaults.py')
    if os.path.exists(filename):
        from pymor.tools.io import file_owned_by_current_user
        if not file_owned_by_current_user(filename):
            raise OSError('Cannot load pyMOR defaults from config file ' + filename
                          + ': not owned by user running Python interpreter')
        print('Loading pyMOR defaults from file ' + filename)
        load_defaults_from_file(filename)

from pymor.core.logger import set_log_format, set_log_levels

set_log_levels()
set_log_format()


@defaults('enabled')
def auto_load_jupyter_extension(enabled=True):
    if not enabled:
        return
    if is_jupyter() and config.HAVE_IPYWIDGETS:
        from IPython import get_ipython
        ip = get_ipython()
        ip.run_line_magic('load_ext', 'pymor.tools.jupyter')

auto_load_jupyter_extension()


from pymor.tools import mpi

if mpi.parallel and mpi.event_loop_settings()['auto_launch']:
    if mpi.rank0:
        import atexit
        @atexit.register
        def quit_event_loop():
            if not mpi.finished:
                mpi.quit()
    else:
        print(f'Rank {mpi.rank}: MPI parallel run detected. Launching event loop ...')
    mpi.launch_event_loop()
    if not mpi.rank0:
        sys.exit(0)
