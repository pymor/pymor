'''
Created on Nov 16, 2012

@author: r_milk01
'''
import unittest
import logging

from pymor.core import BaseInterface
from pymor.core import exceptions

class StupidInterface(BaseInterface):
    '''I am a stupid Interface'''

    @BaseInterface.contract
    @BaseInterface.abstractmethod
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

class BrilliantInterface(BaseInterface):
    '''I am a brilliant Interface'''

    @BaseInterface.contract
    @BaseInterface.abstractmethod
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

    @BaseInterface.contract
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
        try:
            b.new = 42
        except exceptions.ConstError as e:
            print e
        b.freeze()
        print(b.level)
        try:
            b.level = 0
        except exceptions.ConstError as e:
            print e
        b.freeze(False)
        b.level = 0
        b.lock(False)
        b.level = 0

    def testContract(self):
        try:
            #b = AverageImplementer()
            b = StupidImplementer()
            #b = DocImplementer()
            b.shout('Wheee\n', 6)
            #b.whisper('Wheee\n', -2)
        except exceptions.ContractNotRespected as e:
            pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()