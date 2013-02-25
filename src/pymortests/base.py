from __future__ import absolute_import, division, print_function
import unittest
import nose
import logging
import os

from pymor.core.interfaces import BasicInterface
from pymor.core import logger
import pymor

class TestBase(unittest.TestCase, BasicInterface):
    
    @classmethod
    def _is_actual_testclass(cls):
        return cls.__name__ != 'TestBase' and not cls.__name__.endswith('Interface') 
    
    '''only my subclasses will set this to True, prevents nose from thinking I'm an actual test'''
    __test__ = _is_actual_testclass
    
def runmodule(name):
    root_logger = logger.getLogger('pymor')
    root_logger.setLevel(logging.ERROR)
    test_logger = logger.getLogger(name)
    test_logger.setLevel(logging.DEBUG)
    config_files = nose.config.all_config_files()
    config_files.append(os.path.join(os.path.dirname(pymor.__file__), '../../setup.cfg'))
    #config defaults to no plugins -> specify defaults...
    manager = nose.plugins.manager.DefaultPluginManager()
    config = nose.config.Config(files=config_files, plugins=manager)
    return nose.core.runmodule(name=name, config=config)