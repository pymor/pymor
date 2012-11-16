'''
Created on Nov 16, 2012

@author: r_milk01
'''
import unittest
import logging

from nose.tools import raises

from pymor.core.interfaces import (BasicInterface, contract, abstractmethod)
from pymor.core import exceptions

class StupidInterface(BasicInterface):
    '''I am a stupid Interface'''

    @contract
    @abstractmethod
    def shout(self, phrase, repeat):
        """
        :type phrase: str
        :type repeat: int,>0
        """
        pass

    def implementors(self):
        '''I'm just here to overwrite my parent func's docstring
        w/o having a decorator
        '''
        pass

class BrilliantInterface(BasicInterface):
    '''I am a brilliant Interface'''

    @contract
    @abstractmethod
    def whisper(self, phrase, repeat):
        """
        :type phrase: str
        :type repeat: int,=1
        """
        pass

class StupidImplementer(StupidInterface):

    def shout(self, phrase, repeat):
        print(phrase*repeat)

class AverageImplementer(StupidInterface, BrilliantInterface):

    def shout(self, phrase, repeat):
        #cannot change docstring here or else
        print(phrase*repeat)

    def whisper(self, phrase, repeat):
        print(phrase*repeat)

class DocImplementer(AverageImplementer):
    """I got my own docstring"""

    @contract
    def whisper(self, phrase, repeat):
        """my interface is stupid, I can whisper a lot more
        Since I'm overwriting an existing contract, I need to be decorated anew.

        :type phrase: str
        :type repeat: int,>0
        """
        print(phrase*repeat)

class FailImplementer(StupidInterface):
    pass

class InterfaceTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testFreeze(self):
        b = AverageImplementer()
        b.level = 43
        b.lock()
        b.level = 41
        with self.assertRaises(exceptions.ConstError):
            b.new = 42
        b.freeze()
        with self.assertRaises(exceptions.ConstError):
            b.level = 0
        b.freeze(False)
        b.level = 0
        b.lock(False)
        b.level = 0

    @raises(exceptions.ContractNotRespected)
    def testContractFail(self):
        AverageImplementer().whisper('Wheee\n', -2)

    def testContractSuccess(self):
        AverageImplementer().shout('Wheee\n', 6)
            

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()