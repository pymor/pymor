#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:39:53 2012

@author: r_milk01
"""

import abc
import logging
import pprint
import types

from decorators import contract, contracts_decorate

class UberMeta(abc.ABCMeta):
    def __init__(cls, name, bases, namespace):
        '''I copy my class docstring if deriving class has none'''
        doc = namespace.get("__doc__", None)
        if not doc:
            for base in cls.__mro__[1:]:
                if base.__doc__:
                    doc = base.__doc__
                    break
        cls.__doc__ = doc
        super(UberMeta, cls).__init__(name, bases, namespace)

    def __new__(cls,classname,bases,classdict):
        '''I copy decoratorions and docstrings from base class methods to deriving classes'''
        for attr,item in classdict.items():
            if isinstance(item, types.FunctionType):
                #first copy docs
                doc = None
                contract_kwargs = None
                for base in bases:
                    base_func = getattr(base, item.__name__, None)
                    #logging.debug()
                    if base_func:
                        doc = getattr(item, '__doc__', None) or getattr(base_func, '__doc__', None)
                        contract_kwargs = getattr(base_func, 'contract_kwargs', dict())
                if doc:
                    item.__doc__ = doc

                if True:#contract_kwargs:
                    #TODO why the rebind?
                    classdict['_H_%s'%attr] = item    # rebind the method
                    contract_kwargs = contract_kwargs or dict()
                    p = contracts_decorate(item,modify_docstring=True,**contract_kwargs) # replace method by wrapper
                    classdict[attr] = p

        return super(UberMeta, cls).__new__(cls, classname, bases, classdict)

class StupidInterface(object):
    '''I am a stupid Interface'''

    __metaclass__ = UberMeta

    @contract
    @abc.abstractmethod
    def shout(self, phrase, repeat):
        """
        :type phrase: str
        :type repeat: int,>0
        """
        pass

class BrilliantInterface(object):
    '''I am a brilliant Interface'''

    __metaclass__ = UberMeta

    @contract
    @abc.abstractmethod
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
        print(phrase*repeat)

    def whisper(self, phrase, repeat):
        print(phrase*repeat)

class DocImplementer(AverageImplementer):
    """I got my own docstring"""

    @contract
    def whisper(self, phrase, repeat):
        """my interface is stupid, I can whisper a lot more
        :type phrase: str
        :type repeat: int,>0
        """
        print(phrase*repeat)

class FailImplementer(StupidInterface):
    pass

def test_contract():
    b = AverageImplementer()
    b = DocImplementer()
    print(help(b))
    b.shout('Wheee\n', 6)
    b.whisper('Wheee\n', 2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import contracts, traceback
    try:
        test_contract()
    except contracts.ContractNotRespected as e:
        logging.error(e.error + traceback.format_exc(e))

    f = FailImplementer()