# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function
import abc
import types
import itertools
import contracts
import copy
import inspect
from types import NoneType

import numpy as np

from pymor.core import decorators, backports, logger
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

        # monkey a new contract into the decorator module so checking for that type at runtime can work
        dname = (cls.__module__ + '.' + name).replace('__main__.', 'main.').replace('.', '_')
        if not dname in decorators.__dict__:
            decorators.__dict__[dname] = contracts.new_contract(dname, lambda x: isinstance(x, cls))

        # all bases except object get the derived class' name appended
        for base in [b for b in bases if b != object]:
            derived = cls
            # mangle the name to the base scope
            attribute = '_%s__implementors' % base.__name__
            if hasattr(base, attribute):
                getattr(base, attribute).append(derived)
            else:
                setattr(base, attribute, [derived])
        cls.logger = logger.getLogger('{}.{}'.format(cls.__module__.replace('__main__', 'pymor'), name))
        abc.ABCMeta.__init__(cls, name, bases, namespace)

    def __new__(cls, classname, bases, classdict):
        '''I copy contract decorations and docstrings from base class methods to deriving classes.
        I also forward "abstract{class|static}method" decorations in the base class to "{class|static}method"
        decorations in the new subclass.
        '''
        if 'init_arguments' in classdict:
            raise ValueError('init_arguments is a reserved class attribute for subclasses of BasicInterface')

        for attr, item in classdict.items():
            if isinstance(item, types.FunctionType):
                # first copy/fixup docs
                item.__doc__ = decorators.fixup_docstring(item.__doc__)
                base_doc = None
                contract_kwargs = dict()
                for base in bases:
                    has_contract = False
                    base_func = getattr(base, item.__name__, None)
                    if base_func:
                        base_doc = getattr(base_func, '__doc__', None)
                        has_contract = getattr(base_func, 'decorated', None) == 'contract'
                        contract_kwargs = getattr(base_func, 'contract_kwargs', contract_kwargs)
                    if base_doc:
                        doc = decorators.fixup_docstring(getattr(item, '__doc__', ''))
                        has_base_contract_docs = decorators.contains_contract(base_doc)
                        has_contract_docs = decorators.contains_contract(doc)
                        if has_base_contract_docs and not has_contract_docs:
                            base_doc += doc
                        elif not has_base_contract_docs and doc is not None:
                            base_doc = doc
                        item.__doc__ = base_doc
                    if has_contract:
                        # TODO why is the rebind necessary?
                        classdict['_H_%s' % attr] = item
                        contract_kwargs = contract_kwargs or dict()
                        p = decorators.contracts_decorate(item, modify_docstring=True, **contract_kwargs)
                        classdict[attr] = p
                    if (hasattr(base_func, "__isabstractstaticmethod__") and
                            getattr(base_func, "__isabstractstaticmethod__")):
                        classdict[attr] = staticmethod(classdict[attr])
                    if (hasattr(base_func, "__isabstractclassmethod__") and
                            getattr(base_func, "__isabstractclassmethod__")):
                        classdict[attr] = classmethod(classdict[attr])

        c = abc.ABCMeta.__new__(cls, classname, bases, classdict)

        # Beware! The following will probably break in python 3 if there are
        # keyword-only arguemnts
        args, varargs, keywords, defaults = inspect.getargspec(c.__init__)
        if varargs:
            raise NotImplementedError
        assert args[0] == 'self'
        c.init_arguments = tuple(args[1:])
        return c


class BasicInterface(object):
    ''' All other interface classes should be a subclass of mine.
    '''

    __metaclass__ = UberMeta
    _locked = False
    _lock_whitelist = set()

    def __init__(self):
        pass

    def __setattr__(self, key, value):
        '''depending on _locked state I delegate the setattr call to object or
        raise an Exception
        '''
        if not self._locked:
            return object.__setattr__(self, key, value)
        elif key.startswith('_') or key in self._lock_whitelist:
            return object.__setattr__(self, key, value)
        else:
            raise ConstError('Changing "%s" is not allowed in locked "%s"' % (key, self.__class__))

    @property
    def locked(self):
        return self._locked

    def lock(self, doit=True, whitelist=None):
        '''Calling me results in subsequent changes to members throwing errors'''
        object.__setattr__(self, '_locked', doit)
        if whitelist is not None:
            object.__setattr__(self, '_lock_whitelist', whitelist)

    def unlock(self):
        object.__setattr__(self, '_locked', False)

    with_arguments = set()

    def with_(self, **kwargs):
        '''Returns a copy with changed attributes.

        Parameters
        ----------
        **kwargs
            Names of attributes to change with their new values. Each attribute name
            has to be contained in `with_arguments`.

        Returns
        -------
        Copy of `self` with changed attributes.
        '''
        if not set(kwargs.keys()) <= self.with_arguments:
            raise ConstError('Changing "{}" using with() is not allowed in {} (only "{}")'.format(
                kwargs.keys(), self.__class__, self.with_arguments))
        c = copy.copy(self)
        locked = c._locked
        self._locked = False
        for k, v in kwargs.iteritems():
            setattr(c, k, v)
        if c._added_attributes is not None:
            c._added_attributes = list(c._added_attributes)
        c._locked = locked
        return c

    def _with_via_init(self, kwargs, new_class=None):
        '''Default implementation for with_ by calling __init__.

        Parameters which are missing in `kwargs` and for which there is no attribute in `self` are set
        to None.
        '''
        my_type = type(self) if new_class is None else new_class
        argnames = inspect.getargspec(my_type.__init__)[0]
        init_args = kwargs
        for arg in argnames:
            if arg == 'self':
                continue
            if arg not in init_args:
                init_args[arg] = getattr(self, arg, None)
        c = my_type(**init_args)
        if self.logging_disabled:
            c.disable_logging()
        return c

    _added_attributes = None
    def add_attributes(self, **kwargs):
        assert not any(hasattr(self, k) for k in kwargs)
        self.__dict__.update(kwargs)
        if self._added_attributes is None:
            self._added_attributes = kwargs.keys()
        else:
            self._added_attributes.extend(kwargs.keys())

    logging_disabled = False
    def disable_logging(self, doit=True):
        locked = self._locked
        self.unlock()
        if doit:
            self.logger = logger.dummy_logger
            self.logging_disabled = True
        else:
            self.logger = type(self).logger
            self.logging_disabled = False
        self.lock(locked)

    def enable_logging(self, doit=True):
        self.disable_logging(not doit)

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

    @classmethod
    def has_interface_name(cls):
        name = cls.__name__
        return name.endswith('Interface')


contract = decorators.contract
abstractmethod = abc.abstractmethod
abstractproperty = abc.abstractproperty

import sys
if sys.version_info >= (3, 1, 0):
    abstractclassmethod_base = abc.abstractclassmethod
    abstractstaticmethod_base = abc.abstractstaticmethod
else:
    # backport path for issue5867
    abstractclassmethod_base = backports.abstractclassmethod
    abstractstaticmethod_base = backports.abstractstaticmethod


class abstractclassmethod(abstractclassmethod_base):
    '''I mark my wrapped function with an additional __isabstractclassmethod__ member,
    where my abstractclassmethod_base sets __isabstractmethod__ = True.
    '''

    def __init__(self, callable_method):
        callable_method.__isabstractclassmethod__ = True
        super(abstractclassmethod, self).__init__(callable_method)


class abstractstaticmethod(abstractstaticmethod_base):
    '''I mark my wrapped function with an additional __isabstractstaticmethod__ member,
    where my abstractclassmethod_base sets __isabstractmethod__ = True.
    '''

    def __init__(self, callable_method):
        callable_method.__isabstractstaticmethod__ = True
        super(abstractstaticmethod, self).__init__(callable_method)

def _calculate_sid(obj, name):
    if hasattr(obj, 'sid'):
        return obj.sid
    else:
        t_obj = type(obj)
        if t_obj in (tuple, list):
            return tuple(_calculate_sid(o, '{}[{}]'.format(name, i)) for i, o in enumerate(obj))
        elif t_obj is dict:
            return tuple((k, _calculate_sid(v, '{}[{}]'.format(name, k))) for k, v in sorted(obj.iteritems()))
        elif t_obj in (NoneType, str, int, float, bool):
            return obj
        elif t_obj is np.ndarray:
            if obj.size < 64:
                return ('array', obj.shape,obj.dtype.str, obj.tostring())
            else:
                raise ValueError('sid calculation faild at large numpy array {}'.format(name))
        else:
            raise ValueError('sid calculation failed at {}={}'.format(name,type(obj)))


class ImmutableMeta(UberMeta):

    def __new__(cls, classname, bases, classdict):
        c = UberMeta.__new__(cls, classname, bases, classdict)
        init_arguments = c.init_arguments
        try:
            for a in c.sid_ignore:
                if a not in init_arguments and a not in ('name', 'caching'):
                    raise ValueError(a)
        except ValueError as e:
            c.logger.warn('sid_ignore contains "{}" which is not an __init__ argument!'.format(e))
        return c


    def __call__(self, *args, **kwargs):
        instance = super(ImmutableMeta, self).__call__(*args, **kwargs)
        if instance.calculate_sid:
            sid_ignore = instance.sid_ignore

            try:
                arg_sids = tuple(_calculate_sid(o, name)
                                 for o, name in itertools.izip(args, instance.init_arguments)
                                 if not name in sid_ignore)
                kwarg_sids = tuple(_calculate_sid(o, k)
                                   for k, o in sorted(kwargs.iteritems())
                                   if k not in sid_ignore)
                instance.sid = (type(instance), arg_sids, kwarg_sids)
            except ValueError as e:
                instance.sid_failure = str(e)
        else:
            instance.sid_failure = 'disabled'

        instance.lock()
        return instance


class ImmutableInterface(BasicInterface):
    __metaclass__ = ImmutableMeta
    calculate_sid = True
    sid_ignore = ('name', 'caching')

    # Unlocking an immutable object will result in the deletion of its sid.
    # However, this will not delete the sids of objects referencing it.
    # You really should not unlock an object unless you really know what
    # you are doing. (One exception might be the modification of a newly
    # created copy of an immutable object.)
    def lock(self, doit=True, whitelist=None):
        super(ImmutableInterface, self).lock(doit, whitelist)
        if not self.locked and hasattr(self, 'sid'):
            del self.sid
            self.sid_failure = 'unlocked'

    def unlock(self):
        super(ImmutableInterface, self).unlock()
        if hasattr(self, 'sid'):
            del self.sid
            self.sid_failure = 'unlocked'
