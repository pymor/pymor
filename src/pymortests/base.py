# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import pprint
import pkgutil
import sys
import numpy.testing as npt

from pymor.core import logger


class TestInterface(object):

    logger = logger.getLogger(__name__)

    def assertIsInstance(self, arg, cls, msg=None):
        assert isinstance(arg, cls)

    def assertTrue(self, arg, msg=None):
        assert arg

    def assertFalse(self, arg, msg=None):
        assert not arg

    def assertIs(self, arg, other, msg=None):
        assert arg is other

    def assertEqual(self, arg, other, msg=None):
        assert arg == other

    def assertNotEqual(self, arg, other, msg=None):
        assert arg != other

    def assertAlmostEqual(self, arg, other, msg=None):
        npt.assert_almost_equal(arg, other)

    def assertGreaterEqual(self, arg, other, msg=None):
        assert arg >= other

    def assertGreater(self, arg, other, msg=None):
        assert arg > other

    def assertLessEqual(self, arg, other, msg=None):
        assert arg <= other

    def assertLess(self, arg, other, msg=None):
            assert arg < other


TestInterface = TestInterface


def _load_all():
    import pymor
    ignore_playground = True
    fails = []
    for _, module_name, _ in pkgutil.walk_packages(pymor.__path__, pymor.__name__ + '.',
                                                   lambda n: fails.append((n, ''))):
        if ignore_playground and 'playground' in module_name:
            continue
        try:
            __import__(module_name, level=0)
        except (TypeError, ImportError) as t:
            fails.append((module_name, t))
    if len(fails) > 0:
        logger.getLogger(__name__).fatal('Failed imports: {}'.format(pprint.pformat(fails)))
        raise ImportError(__name__)


def SubclassForImplemetorsOf(InterfaceType):
    '''A decorator that dynamically creates subclasses of the decorated base test class
    for all implementors of a given Interface
    '''
    try:
        _load_all()
    except ImportError:
        pass

    def decorate(TestCase):
        '''saves a new type called cname with correct bases and class dict in globals'''
        import pymor.core.dynamic
        test_types = set([T for T in InterfaceType.implementors(True) if not(T.has_interface_name()
                                                                             or issubclass(T, TestInterface))])
        for Type in test_types:
            cname = 'Test_{}_{}'.format(Type.__name__, TestCase.__name__.replace('Interface', ''))
            pymor.core.dynamic.__dict__[cname] = type(cname, (TestCase,), {'Type': Type})
        return TestCase
    return decorate


def runmodule(filename):
    import pytest
    sys.exit(pytest.main(sys.argv[1:] + [filename]))
