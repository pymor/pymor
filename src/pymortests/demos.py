# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pkgutil
import pymordemos
import runpy
import sys
import pytest

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
             ('thermalblock', [2, 2, 3, 5]), ('thermalblock', ['--greedy-without-estimator', 2, 2, 3, 5]),
             ('thermalblock_gui', ['--testing', 2, 2, 3, 5]),
             ('thermalblock', ['--alg=pod', 2, 2, 3, 5]),
             ('thermalblock', ['--alg=adaptive_greedy', 2, 2, 10, 30]),
             ('parabolic', [1]),
             ('parabolic', ['--rect', 1]),
             ('parabolic', ['--fv', 1]),
             ('parabolic', ['--rect', '--fv', 1]))
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
    try:
        ret = _run(module, args)
        # TODO find a better/tighter assert/way to run the code
        assert ret is not None
    except ImportError as ie:
        assert _is_failed_import_ok(ie)
    stop_gui_processes()


if __name__ == "__main__":
    runmodule(filename=__file__)
