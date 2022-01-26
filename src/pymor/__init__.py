# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import os
import platform
import sys


if sys.version_info.major < 3:
    raise RuntimeError('pyMOR requires Python 3.6 or newer')
elif sys.version_info.major == 3:
    if sys.version_info.minor < 6:
        raise RuntimeError('pyMOR requires Python 3.6 or newer.')
    if sys.version_info.minor == 6 and platform.python_implementation() != 'CPython':
        # dicts are only guaranteed to be insertion ordered for Python 3.7 or newer.
        # For CPython this is already the case for Python 3.6.
        raise RuntimeError('pyMOR requires Python 3.7 or newer for non-CPython interpreters.')


def _init_mpi():
    """provides a way to manually set the thread init mode for MPI if necessary.
    Needs to happen as early as possible, otherwise mpi4py might auto-init somewhere else.


    """

    try:
        import mpi4py
    except ImportError:
        return

    finalize = os.environ.get('PYMOR_MPI_FINALIZE', mpi4py.rc.finalize if mpi4py.rc.finalize is not None else True)
    mpi4py.rc(initialize=False, finalize=finalize)
    from mpi4py import MPI
    if not MPI.Is_initialized():
        required_level = int(os.environ.get('PYMOR_MPI_INIT_THREAD', MPI.THREAD_MULTIPLE))
        supported_lvl = MPI.Init_thread(required_level)
        if supported_lvl < required_level:
            print(f'MPI does support threading level {required_level}, running with {supported_lvl} instead', flush=True)

    try:
        # this solves sporadic mpi calls happening after finalize
        import petsc4py
        petsc4py.init()
    except ImportError:
        return

_init_mpi()

from pymor.core.config import config
from pymor.core.defaults import load_defaults_from_file


import pymor.version as _version
revstring = _version.get_versions()['version']
__version__ = str(revstring)

if 'PYMOR_DEFAULTS' in os.environ:
    filename = os.environ['PYMOR_DEFAULTS']
    if filename in ('', 'NONE'):
        print('Not loading any pyMOR defaults from config file')
    else:
        for fn in filename.split(':'):
            if not os.path.exists(fn):
                raise IOError('Cannot load pyMOR defaults from file ' + fn)
            print('Loading pyMOR defaults from file ' + fn + ' (set by PYMOR_DEFAULTS)')
            load_defaults_from_file(fn)
else:
    filename = os.path.join(os.getcwd(), 'pymor_defaults.py')
    if os.path.exists(filename):
        from pymor.tools.io import file_owned_by_current_user
        if not file_owned_by_current_user(filename):
            raise IOError('Cannot load pyMOR defaults from config file ' + filename
                          + ': not owned by user running Python interpreter')
        print('Loading pyMOR defaults from file ' + filename)
        load_defaults_from_file(filename)

from pymor.core.logger import set_log_levels, set_log_format
set_log_levels()
set_log_format()

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
        import sys
        sys.exit(0)
