# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import unittest
import nose
import logging
import os
import pprint
import pkgutil

from pymor.core.interfaces import BasicInterface
from pymor.core import logger


class PymorTestProgram(nose.core.TestProgram):
    pass


class PymorTestSelector(nose.selector.Selector):

    def __init__(self, *args, **kwargs):
        super(PymorTestSelector, self).__init__(*args, **kwargs)
        self._skip_grid = 'PYMOR_NO_GRIDTESTS' in os.environ

    def wantDirectory(self, dirname):
        return 'src' in dirname

    def wantFile(self, filename):
        if self._skip_grid and 'grid' in filename:
            return False
        return filename.endswith('.py') and ('pymortests' in filename or
                                             'dynamic' in filename)

    def wantModule(self, module):
        parts = module.__name__.split('.')
        return 'pymortests' in parts or 'pymor' in parts

    def wantClass(self, cls):
        ret = super(PymorTestSelector, self).wantClass(cls)
        if hasattr(cls, 'has_interface_name'):
            return ret and not cls.has_interface_name()
        return ret


class TestBase(unittest.TestCase, BasicInterface):

    @classmethod
    def _is_actual_testclass(cls):
        return cls.__name__ != 'TestBase' and not cls.has_interface_name()

    '''only my subclasses will set this to True, prevents nose from thinking I'm an actual test'''
    __test__ = _is_actual_testclass


def _load_all():
    import pymor
    fails = []
    for _, module_name, _ in pkgutil.walk_packages(pymor.__path__, pymor.__name__ + '.',
                                                   lambda n: fails.append((n, ''))):
        try:
            __import__(module_name, level=0)
        except TypeError, t:
            fails.append((module_name, t))
    if len(fails) > 0:
        logger.getLogger(__name__).fatal('Failed imports: {}'.format(pprint.pformat(fails)))
        raise ImportError(__name__)


def SubclassForImplemetorsOf(InterfaceType):
    '''A decorator that dynamically creates subclasses of the decorated base test class
    for all implementors of a given Interface
    '''
    _load_all()

    def decorate(TestCase):
        '''saves a new type called cname with correct bases and class dict in globals'''
        import pymor.core.dynamic
        for Type in set([T for T in InterfaceType.implementors(True) if (not T.has_interface_name()
                                                                         and not issubclass(T, TestBase))]):
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
    _load_all()

    def decorate(TestCase):
        '''saves a new type called cname with correct bases and class dict in globals'''
        import pymor.core.dynamic
        for GridType in set([T for T in InterfaceType.implementors(True) if not T.has_interface_name()]):
            cname = '{}_{}'.format(GridType.__name__, TestCase.__name__.replace('Interface', ''))
            pymor.core.dynamic.__dict__[cname] = type(cname, (TestCase,), {'grids': GridType.test_instances(),
                                                      '__test__': True})
            assert len(pymor.core.dynamic.__dict__[cname].grids) > 0
        return TestCase
    return decorate

def _setup(name='pymor'):
    root_logger = logger.getLogger(name)
    root_logger.setLevel(logging.ERROR)
    test_logger = logger.getLogger(name)
    test_logger.setLevel(logging.DEBUG)  # config_files.append(os.path.join(os.path.dirname(pymor.__file__), '../../setup.cfg'))
    # config defaults to no plugins -> specify defaults...
    manager = nose.plugins.manager.DefaultPluginManager()
    config_files = nose.config.all_config_files()
    config = nose.config.Config(files=config_files, plugins=manager)
    config.exclude = []
    selector = PymorTestSelector(config=config)
    loader = nose.loader.defaultTestLoader(config=config, selector=selector)
    cli = [__file__, '-vv', '-d']
    return cli, loader, config

def suite():
    cli, loader, cfg = _setup()
    prog = nose.core.TestProgram(argv=cli, testLoader=loader, config=cfg, module='pymortests')
    prog.createTests()
    return prog.suite

def runmodule(name):
    cli, loader, cfg = _setup(name)
    return nose.core.runmodule(name=name, config=cfg, testLoader=loader, argv=cli)
