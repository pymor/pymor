# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import os
import pkgutil
import pymordemos
import runpy
import sys
import pytest
from tempfile import mkdtemp
import shutil

from pymortests.base import runmodule
from pymor.gui.gl import HAVE_PYSIDE
from pymor.gui.qt import stop_gui_processes

DEMO_ARGS = (('elliptic', [0, 0, 0, 0]), ('elliptic', [1, 2, 0, 3]), ('elliptic', ['--rect', 1, 2, 0, 3]),
             ('elliptic', [0, 0, 2, 1]), ('elliptic', ['--fv', 0, 0, 0, 0]), ('elliptic', ['--fv', 1, 2, 0, 3]),
             ('elliptic', ['--fv', '--rect', 1, 2, 0, 3]), ('elliptic', ['--fv', 0, 0, 2, 1]),
             ('burgers', ['--num-flux=lax_friedrichs', '0.1']), ('burgers', ['--num-flux=engquist_osher', '0.1']),
             ('burgers_ei', [1, 2, 2, 5, 2, 5]), ('burgers', ['--num-flux=simplified_engquist_osher', '0.1']),
             ('elliptic2', [1, 20]), ('elliptic2', ['--fv', 1, 20]),
             ('elliptic_unstructured', [6., 16, 1e-1]),
             ('elliptic_oned', [1, 20]), ('elliptic_oned', ['--fv', 1, 20]),
             ('thermalblock', ['--plot-solutions', '--plot-err', '--plot-error-sequence', 2, 2, 3, 5]),
             ('thermalblock', ['--fenics', 2, 2, 3, 5]),
             ('thermalblock', ['--greedy-without-estimator', 3, 1, 2, 5]),
             ('thermalblock_gui', ['--testing', 2, 2, 3, 5]),
             ('thermalblock', ['--alg=pod', 2, 2, 3, 5]),
             ('thermalblock', ['--alg=adaptive_greedy', 2, 2, 10, 30]),
             ('thermalblock', ['--alg=naive', '--reductor=traditional', 2, 2, 10, 30]),
             ('thermalblock_adaptive', [10]),
             ('thermalblock_adaptive', ['--visualize-refinement', 10]),
             ('thermalblock_simple', ['pymor', 'naive', 2, 10, 10]),
             ('thermalblock_simple', ['fenics', 'greedy', 2, 10, 10]),
             ('parabolic', [1]),
             ('parabolic', ['--rect', 1]),
             ('parabolic', ['--fv', 1]),
             ('parabolic', ['--rect', '--fv', 1]),
             ('parabolic_mor', ['pymor', 'greedy', 2, 3, 1]),
             ('parabolic_mor', ['pymor', 'pod', 2, 3, 1]),
             ('parabolic_mor', ['fenics', 'adaptive_greedy', 2, 3, 1]))
DEMO_ARGS = [('pymordemos.{}'.format(a), b) for (a, b) in DEMO_ARGS]
# DEMO_ARGS = [('pymor.playground.demos.remote_thermalblock', ['--local', '-e',2, 2, 3, 5])]


def _run(module, args):
    sys.argv = [module] + [str(a) for a in args]
    return runpy.run_module(module, init_globals=None, run_name='__main__', alter_sys=True)


@pytest.fixture(params=DEMO_ARGS)
def demo_args(request):
    return request.param


def _is_failed_import_ok(error):
    if error.message == 'cannot visualize: import of PySide failed':
        return not HAVE_PYSIDE
    return False


def test_demos(demo_args):
    module, args = demo_args

    import sys
    sys._called_from_test = True

    def nop(*args, **kwargs):
        pass

    try:
        from matplotlib import pyplot
        pyplot.show = nop
    except ImportError:
        pass
    try:
        import dolfin
        dolfin.plot = nop
        dolfin.interactive = nop
    except ImportError:
        pass
    try:
        ret = _run(module, args)
        # TODO find a better/tighter assert/way to run the code
        assert ret is not None
    except ImportError as ie:
        assert _is_failed_import_ok(ie), ie
    finally:
        stop_gui_processes()
        from pymor.parallel.default import _cleanup
        _cleanup()


def test_analyze_pickle1():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['histogram', '--error-norm=h1_0_semi', os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_analyze_pickle2():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['histogram', '--detailed=' + os.path.join(d, 'data_detailed'),
                    os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_analyze_pickle3():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['convergence', '--error-norm=h1_0_semi', os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_analyze_pickle4():
    d = mkdtemp()
    try:
        test_demos(('pymordemos.thermalblock', ['--pickle=' + os.path.join(d, 'data'), 2, 2, 2, 10]))
        test_demos(('pymordemos.analyze_pickle',
                   ['convergence', '--detailed=' + os.path.join(d, 'data_detailed'),
                    os.path.join(d, 'data_reduced'), 10]))
    finally:
        shutil.rmtree(d)


def test_thermalblock_ipython(demo_args):
    if demo_args[0] != 'pymordemos.thermalblock':
        return
    from pymor.tools import mpi
    if mpi.parallel:  # simply running 'ipcluster start' (without any profile) does not seem to work
        return        # when running under mpirun, so we do not test this combination for now
    try:
        test_demos((demo_args[0], ['--ipython-engines=2'] + demo_args[1]))
    finally:
        import time     # there seems to be no way to shutdown the IPyhton cluster s.t. a new
        time.sleep(10)  # cluster can be started directly afterwards, so we have to wait ...


if __name__ == "__main__":
    runmodule(filename=__file__)
