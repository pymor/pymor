# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import unittest
import os
import pprint
import pkgutil
import sys
import importlib
import pytest

from pymor.core.interfaces import BasicInterface
from pymor.core import logger


class TestBase(unittest.TestCase, BasicInterface):

    @classmethod
    def _is_actual_testclass(cls):
        return cls.__name__ != 'TestBase' and not cls.has_interface_name()

    '''only my subclasses will set this to True, maybe prevents pytest from thinking I'm an actual test'''
    __test__ = _is_actual_testclass


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
        test_types = set([T for T in InterfaceType.implementors(True) if (not T.has_interface_name()
                                                                          and not issubclass(T, TestBase))])
        for Type in test_types:
            cname = '{}_{}'.format(Type.__name__, TestCase.__name__.replace('Interface', ''))
            pymor.core.dynamic.__dict__[cname] = type(cname, (TestCase,), {'__test__': True, 'Type': Type})
        return TestCase
    return decorate


class GridClassTestInterface(TestBase):
    pass


def GridSubclassForImplemetorsOf(InterfaceType):
    '''A decorator that dynamically creates subclasses of the decorated base test class
    for all implementors of a given Interface
    '''
    try:
        _load_all()
    except ImportError:
        pass

    def _getType(name):
        module = name[0:name.rfind('.')]
        classname = name[name.rfind('.') + 1:]
        importlib.import_module(module)
        return sys.modules[module].__dict__[classname]

    def decorate(TestCase):
        '''saves a new type called cname with correct bases and class dict in globals'''
        import pymor.core.dynamic
        if 'PYMOR_GRID_TYPE' in os.environ:
            test_types = [_getType(os.environ['PYMOR_GRID_TYPE'])]
        else:
            test_types = set([T for T in InterfaceType.implementors(True) if not T.has_interface_name()])
        for GridType in test_types:
            cname = '{}_{}'.format(GridType.__name__, TestCase.__name__.replace('Interface', ''))
            pymor.core.dynamic.__dict__[cname] = type(cname, (TestCase,), {'grids': GridType.test_instances(),
                                                      '__test__': True})
            assert len(pymor.core.dynamic.__dict__[cname].grids) > 0
        return TestCase
    return decorate


def runmodule(filename):
    return pytest.main(filename)
