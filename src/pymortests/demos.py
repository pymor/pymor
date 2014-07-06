# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import pkgutil
import pymordemos
import runpy
import sys
import pytest

from pymortests.base import TestInterface, runmodule

DEMO_ARGS = (('cg', [0, 0, 0]), ('cg', [1, 2, 3]),
             ('burgers', ['0.1']), ('burgers_ei', [0.9, 1.1, 2, 5, 3]),
             ('cg2', [1, 20, 0]), ('cg_oned', [1, 20, 0]),
             ('thermalblock', ['-e',2, 2, 3, 5]), ('thermalblock', [2, 2, 3, 5]),
             ('thermalblock_gui', [2, 2, 3, 5]), ('thermalblock_pod', [2, 2, 3, 5]))

def _run(module, args):
    sys.argv = [module] + [str(a) for a in args]
    return runpy.run_module(module, init_globals=None, run_name='__main__', alter_sys=True)


@pytest.fixture(params=DEMO_ARGS)
def demo_args(request):
    return request.param


def test_demos(demo_args):
    for short, args in DEMO_ARGS:
        module = 'pymordemos.{}'.format(short)
        ret = _run(module, args)
        #TODO find a better/tighter assert
        assert ret is not None


def test_demos_tested():
    shorts = []
    for _, module_name, _ in pkgutil.walk_packages(pymordemos.__path__, pymordemos.__name__ + '.'):
        try:
            foo = __import__(module_name)
            short = module_name[len('pymordemos.'):]
            shorts.append(short)
        except (TypeError, ImportError):
            pass
    tested = [f[0] for f in DEMO_ARGS]
    for short in shorts:
        assert short in tested


if __name__ == "__main__":
    runmodule(filename=__file__)