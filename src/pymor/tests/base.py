
import unittest

from pymor.core.interfaces import BasicInterface

class TestBase(unittest.TestCase, BasicInterface):
    
    @classmethod
    def _is_actual_testclass(cls):
        return cls.__name__ != 'TestBase' and not cls.__name__.endswith('Interface') 
    
    '''only my subclasses will set this to True, prevents nose from thinking I'm an actual test'''
    __test__ = _is_actual_testclass
    
    