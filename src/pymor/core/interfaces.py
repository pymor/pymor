#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:39:53 2012

@author: r_milk01
"""

import abc
import types
import itertools
import contracts

from pymor.core import decorators 
from pymor.core.exceptions import ConstError


class UberMeta(abc.ABCMeta):
    def __init__(cls, name, bases, namespace):
        '''I copy my class docstring if deriving class has none. I tell base classes when I derive
        a new class from them. I publish a new contract type for each new class I create. 
        '''
        doc = namespace.get("__doc__", None)
        if not doc:
            for base in cls.__mro__[1:]:
                if base.__doc__:
                    doc = base.__doc__
                    break
        cls.__doc__ = doc
        
        #monkey a new contract into the decorator module so checking for that type at runtime can work
        import logging, pprint
        logging.error('CLS %s -- %s', cls, name)
        decorators.__dict__[name] = contracts.new_contract(name, lambda x: isinstance(x, cls))

        #all bases except object get the derived class' name appended      
        for base in [b for b in bases if b != object]:
            derived = cls
            #mangle the name to the base scope
            attribute = '_%s__implementors'%base.__name__
            if hasattr(base, attribute):
                getattr(base, attribute).append(derived)
            else:
                setattr(base, attribute, [derived])
        super(UberMeta, cls).__init__(name, bases, namespace)

    def __new__(cls,classname,bases,classdict):
        '''I copy contract decorations and docstrings from base class methods to deriving classes.
        I also forward "abstract{class|static}method" decorations in the base class to "{class|static}method" 
        decorations in the new subclass. 
        '''
        for attr,item in classdict.items():
            if isinstance(item, types.FunctionType):
                #first copy docs
                base_doc = None
                contract_kwargs = dict()
                for base in bases:
                    has_contract = False
                    base_func = getattr(base, item.__name__, None)
                    if base_func:
                        base_doc = getattr(base_func, '__doc__', None)
                        has_contract = getattr(base_func,'decorated', None) == 'contract'
                        contract_kwargs = getattr(base_func, 'contract_kwargs', contract_kwargs)
                    if base_doc:
                        doc = getattr(item, '__doc__', '')
                        has_base_contract_docs = decorators.contains_contract(base_doc)
                        has_contract_docs = decorators.contains_contract(doc)
                        if has_base_contract_docs and not has_contract_docs:
                            base_doc += doc
                        elif not has_base_contract_docs and doc is not None:
                            base_doc = doc
                        item.__doc__ = base_doc
                    if has_contract:
                        #TODO why is the rebind necessary?
                        classdict['_H_%s'%attr] = item
                        contract_kwargs = contract_kwargs or dict()
                        p = decorators.contracts_decorate(item,modify_docstring=True,**contract_kwargs)
                        classdict[attr] = p
                    if (hasattr(base_func, "__isabstractstaticmethod__") and 
                        getattr(base_func, "__isabstractstaticmethod__")):
                        classdict[attr] = staticmethod(classdict[attr])
                    if (hasattr(base_func, "__isabstractclassmethod__") and 
                        getattr(base_func, "__isabstractclassmethod__")):
                        classdict[attr] = classmethod(classdict[attr])
                        


        return super(UberMeta, cls).__new__(cls, classname, bases, classdict)


class BasicInterface(object):
    ''' All other interface classes should be a subclass of mine.
    '''

    __metaclass__ = UberMeta
    _locked = False
    _frozen = False
    
    def __setattr__(self, key, value):
        '''depending on _locked nad _frozen state I delegate the setattr call to object or
        raise an Exception
        '''
        if not self._locked:
            return object.__setattr__(self, key, value)

        if self.__dict__.has_key(key):
            if self._frozen:
                raise ConstError('Changing "%s" is not allowed in "%s"' % (key, self.__class__))
            return object.__setattr__(self, key, value)
        else:
            raise ConstError('Won\'t add "%s" to locked "%s"' % (key, self.__class__))

    def lock(self, doit=True):
        '''Calling me results in subsequent adding of members throwing errors'''
        object.__setattr__(self, '_locked', doit)

    def freeze(self, doit=True):
        '''Calling me results in subsequent changes to members throwing errors'''
        object.__setattr__(self, '_frozen', doit)

    @classmethod    
    def implementors(cls, descend=False):
        '''I return a, potentially empty, list of my subclass-objects. 
        If descend is True I traverse my entire subclass hierarchy and return a flattened list.
        '''
        if not hasattr(cls, '_%s__implementors' % cls.__name__):
            return []
        level = getattr(cls, '_%s__implementors' % cls.__name__)       
        if not descend:
            return level
        subtrees = itertools.chain.from_iterable([sub.implementors() for sub in level if sub.implementors() != []]) 
        level.extend(subtrees)
        return level
    
    @classmethod    
    def implementor_names(cls, descend=False):
        '''For convenience I return a list of my implementor names instead of class objects'''
        return [c.__name__ for c in cls.implementors(descend)]

contract = decorators.contract
abstractmethod = abc.abstractmethod

import sys
if sys.version_info >= (3,1,0):
    abstractclassmethod_base = abc.abstractclassmethod
    abstractstaticmethod_base = abc.abstractstaticmethod
else:
    #backport path for issue5867
    import backports
    abstractclassmethod_base = backports.abstractclassmethod
    abstractstaticmethod_base = backports.abstractstaticmethod
    
    
class abstractclassmethod(abstractclassmethod_base):
    '''I mark my wrapped function with an additional __isabstractclassmethod__ member,
    where my abstractclassmethod_base sets __isabstractmethod__ = True.
    '''
    
    def __init__(self, callable):
        callable.__isabstractclassmethod__ = True
        super(abstractclassmethod, self).__init__(callable)


class abstractstaticmethod(abstractstaticmethod_base):
    '''I mark my wrapped function with an additional __isabstractstaticmethod__ member,
    where my abstractclassmethod_base sets __isabstractmethod__ = True.
    '''
    
    def __init__(self, callable):
        callable.__isabstractstaticmethod__ = True
        super(abstractstaticmethod, self).__init__(callable)