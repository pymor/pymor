# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2017 pyMOR developers and contributors. All rights reserved.
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
    import sys

    # ensure that FEniCS visualization does nothing on all MPI ranks
    def monkey_dolfin():
        def nop(*args, **kwargs):
            pass
        try:
            import dolfin
            dolfin.plot = nop
            dolfin.interactive = nop
        except ImportError:
            pass
    mpi.call(monkey_dolfin)

    def disable_ngsolve():
        from pymor.core.config import config
        config.HAVE_NGSOLVE = False
        import os
        try:
            del os.environ['DOCKER_NGSOLVE']
        except KeyError:
            pass
    mpi.call(disable_ngsolve)

    runpy.run_module('pymortests.demos', init_globals=None, run_name='__main__', alter_sys=True)
