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

from pymor.core.decorators import (contract, contracts_decorate, contains_contract,
                        )
from pymor.core.exceptions import ConstError


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
        '''I copy contract decorations and docstrings from base class methods to deriving classes'''
        for attr,item in classdict.items():
            if isinstance(item, types.FunctionType):
                #first copy docs
                base_doc = None
                contract_kwargs = dict()
                for base in bases:
                    has_contract = False
                    base_func = getattr(base, item.__name__, None)
                    #logging.debug()
                    if base_func:
                        base_doc = getattr(base_func, '__doc__', None)
                        has_contract = getattr(base_func,'decorated', None) == 'contract'
                        contract_kwargs = getattr(base_func, 'contract_kwargs', contract_kwargs)
                    if base_doc:
                        doc = getattr(item, '__doc__', '')
                        has_base_contract_docs = contains_contract(base_doc)
                        has_contract_docs = contains_contract(doc)
                        if has_base_contract_docs and not has_contract_docs:
                            base_doc += doc
                        elif not has_base_contract_docs and doc is not None:
                            base_doc = doc
                        item.__doc__ = base_doc
                    if has_contract:
                        #TODO why the rebind?
                        classdict['_H_%s'%attr] = item    # rebind the method
                        contract_kwargs = contract_kwargs or dict()
                        p = contracts_decorate(item,modify_docstring=True,**contract_kwargs) # replace method by wrapper
                        classdict[attr] = p


        return super(UberMeta, cls).__new__(cls, classname, bases, classdict)


class UberInterface(object):
    ''' All other interface classes should be a subclass of mine.
    '''

    __metaclass__ = UberMeta
    _locked = False
    _frozen = False

    def __setattr__(self, key, value):
        if not self._locked:
            return object.__setattr__(self, key, value)

        if self.__dict__.has_key(key):
            if self._frozen:
                raise ConstError('Changing "%s" is not allowed in "%s"' % (key, self.__class__))
            return object.__setattr__(self, key, value)
        else:
            raise ConstError('Won\'t add "%s" to locked "%s"' % (key, self.__class__))

    def lock(self, doit=True):
        '''Calling me results in subsequent changes to members throwing errors'''
        object.__setattr__(self, '_locked', doit)

    def freeze(self, doit=True):
        '''Calling me results in subsequent changes to members throwing errors'''
        object.__setattr__(self, '_frozen', doit)

    def implementors(self):
        '''I do nothing yet'''
        pass

class StupidInterface(UberInterface):
    '''I am a stupid Interface'''

    @contract
    @abc.abstractmethod
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

class BrilliantInterface(UberInterface):
    '''I am a brilliant Interface'''

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

def test_contract():
    #b = AverageImplementer()
    b = StupidImplementer()
    #b = DocImplementer()
    print(help(b))
    b.shout('Wheee\n', 6)
    #b.whisper('Wheee\n', -2)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    import contracts, traceback
    try:
        #test_contract()
        pass
    except contracts.ContractNotRespected as e:
        logging.error(e.error + traceback.format_exc(e))

    #f = FailImplementer()
    b = AverageImplementer()
    logging.basicConfig(level=logging.DEBUG)
    b.level = 43
    b.lock()
    b.level = 41
    try:
        b.new = 42
    except Exception as e:
        print e
    b.freeze()
    print(b.level)
    try:
        b.level = 0
    except Exception as e:
        print e
    b.freeze(False)
    b.level = 0
    b.lock(False)
    b.level = 0
