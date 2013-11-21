# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

'''This module provides base classes from which most classes in pyMOR inherit.

The purpose of these classes is to provide some common functionality for
all objects in pyMOR. The most notable features provided by :class:`BasicInterface`
are the following:

    1. :class:`BasicInterface` sets class :class:`UberMeta` as metaclass
       which itself inherits from :class:`abc.ABCMeta`, so it is possible
       to define interface classes with abstract methods using the
       :func:`abstractmethod` decorator. There are also decorators for
       abstract class methods, static methods, and properties.
    2. Using metaclass magic, each _class_ deriving from :class:`BasicInterface`
       comes with its own logger instance accessible through its `logger`
       attribute. The logger prefix is automatically set to the class name.
    3. Logging can be disabled and reenabled for each _instance_ using the
       :meth:`BasicInterface.disable_logging` and :meth:`BasicInterface.enable_logging`
       methods.
    4. An instance can be made immutable using :meth:`BasicInterface.lock`.
       If an instance is locked, each attempt to change one of its attributes
       raises an exception. Private attributes (of the form `_name`) are exempted
       from this rule. Locked instances can be unlocked again using
       :meth:`BasicInterface.unlock`.
    5. :meth:`BasicInterface.with_` can be used to create a copy of an instance with
       some changed attributes. E.g. ::

           obj.with_(a=x, b=y)

       creates a copy with the `a` and `b` attributes of `obj` set to `x` and `y`.
       Note that in general `a` and `b` do not necessarily have to corresond to
       class attributes of `obj`; it is up to the implementor to interpret the
       provided arguments. :attr:`BasicInterface.with_arguments` holds the
       set of allowed arguments.
       :class:`BasicInterface` provides a default implementation of `with_` which
       works as follows:

           - The argument names of the classes  `__init__` method are looked up.
             If the instance has an attribute of the same name for each `__init__`
             argument, `with_arguments` returns the argument names of `__init__`,
             otherwise an empty set is returned and the `with_` functionality is
             disabled.
           - If the above condition is satisfied, a call to `with_` results in
             the creation of a new instance where the arguments of `with_` are
             passed through to `__init__`. The missing `__init__` arguments
             are taken from the corresponding instance attributes.

    6. :meth:`BasicInterface.uid` provides a unique id for each instance. While
       `id(obj)` is only guaranteed to be unique among all living python objects,
       :meth:`BasicInterface.uid` will be (almost) unique among all pyMOR objects
       that have ever existed. This is achieved by building the id from a uuid4
       which is newly created for each pyMOR run and a counter which is increased
       for any object that requests an uid.
       This functionality is implemented using :class:`UIDProvider`.


:class:`ImmutableMeta` derives from :class:`BasicInterface` and adds the following
functionality:

    1. Using more metaclass magic, each instance which derives from
       :class:`ImmutableInterface` is locked after its `__init__` method has returned.
    2. If possible, a unique state id for the instance is calculated and stored as
       `sid` attribute. If sid calculation fails, `sid_failure` is set to a string
       giving a reason for the failure.
       The sid is constructed as a tuple containing:

           - the class of the instance
           - for each `__init__` argument its name and

               - its `sid` if it has one
               - its value if it is an instance of `NoneType`, `str`, `int`, `float` or `bool`
               - its value if it is a numpy array of short size

             For `tuple`, `list` or `dict` instance, the calculation is done by recursion.
             If none of these cases apply, sid calculation fails.

       Note that a sid contains only object references to the sids of the provided `__init__`
       arguments. This structure is preserved by pickling resulting in relatively short
       string represenations of the sid.
    3. sid generation (with all its overhead) can be disabled by setting
       :attr:`ImmutableInterface.calculate_sid` to `False`.
    4. :attr:`ImmutableInterface.sid_ignore` can be set to a tuple of `__init__`
       argument names, which should be excluded from sid calculation.
    5. sid generation can be disabled completely in pyMOR by calling
       :func:`disable_sid_generation`. It can be activated again by calling
       :func:`enable_sid_generation`.
'''

from __future__ import absolute_import, division, print_function
import abc
import types
import itertools
import contracts
import inspect
from types import NoneType

import numpy as np

from pymor.core import decorators, backports, logger
from pymor.core.exceptions import ConstError


class UIDProvider(object):
    def __init__(self):
        self.counter = 0
        import uuid
        self.prefix = '{}_'.format(uuid.uuid4())

    def __call__(self):
        uid = self.prefix + str(self.counter)
        self.counter += 1
        return uid

uid_provider = UIDProvider()


class UberMeta(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        '''I tell base classes when I derive a new class from them. I publish
        a new contract type for each new class I create.
        '''
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

    def __init__(self):
        pass

    def __setattr__(self, key, value):
        '''depending on _locked state I delegate the setattr call to object or
        raise an Exception
        '''
        if not self._locked or key[0] == '_':
            return object.__setattr__(self, key, value)
        else:
            raise ConstError('Changing "%s" is not allowed in locked "%s"' % (key, self.__class__))

    @property
    def locked(self):
        return self._locked

    def lock(self, doit=True):
        '''Calling me results in subsequent changes to members throwing errors'''
        object.__setattr__(self, '_locked', doit)

    def unlock(self):
        object.__setattr__(self, '_locked', False)

    @property
    def with_arguments(self):
        init_arguments = self.init_arguments
        for arg in init_arguments:
            if not hasattr(self, arg):
                self._with_arguments_error = "Instance does not have attribute for __init__ argument '{}'".format(arg)
                return set()
        return init_arguments

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
        with_arguments = self.with_arguments      # ensure that property is called first
        if hasattr(self, '_with_arguments_error'):
            raise ConstError('Using with_ is not possible because of the following Error: '
                             + self._with_arguments_error)
        if not set(kwargs.keys()) <= with_arguments:
            raise ConstError('Changing "{}" using with() is not allowed in {} (only "{}")'.format(
                kwargs.keys(), self.__class__, self.with_arguments))
        return self._with_via_init(kwargs)

    def _with_via_init(self, kwargs, new_class=None):
        '''Default implementation for with_ by calling __init__.

        Parameters which are missing in `kwargs` and for which there is no attribute in `self` are set
        to None.
        '''
        my_type = type(self) if new_class is None else new_class
        init_args = kwargs
        for arg in self.init_arguments:
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

    _uid = None

    @property
    def uid(self):
        if self._uid is None:
            self._uid = uid_provider()
        return self._uid


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
                return ('array', obj.shape, obj.dtype.str, obj.tostring())
            else:
                raise ValueError('sid calculation faild at large numpy array {}'.format(name))
        else:
            raise ValueError('sid calculation failed at {}={}'.format(name, type(obj)))


def inject_sid(obj, context, *args):
    try:
        sid = tuple((context, tuple(_calculate_sid(o, i) for i, o in enumerate(args))))
        obj.sid = sid
        ImmutableMeta.sids_created += 1
    except ValueError as e:
        obj.sid_failure = str(e)

    if isinstance(obj, BasicInterface):
        obj.lock()
    elif isinstance(obj, np.ndarray):
        obj.flags.writable = False


def disable_sid_generation():
    if hasattr(ImmutableMeta, '__call__'):
        del ImmutableMeta.__call__


def enable_sid_generation():
    ImmutableMeta.__call__ = ImmutableMeta._call


class ImmutableMeta(UberMeta):

    sids_created = 0
    init_arguments_never_warn = ('name', 'caching')

    def __new__(cls, classname, bases, classdict):
        c = UberMeta.__new__(cls, classname, bases, classdict)
        init_arguments = c.init_arguments
        try:
            for a in c.sid_ignore:
                if a not in init_arguments and a not in ImmutableMeta.init_arguments_never_warn:
                    raise ValueError(a)
        except ValueError as e:
            c.logger.warn('sid_ignore contains "{}" which is not an __init__ argument!'.format(e))
        return c

    def _call(self, *args, **kwargs):
        instance = super(ImmutableMeta, self).__call__(*args, **kwargs)
        if instance.calculate_sid:
            try:
                kwargs.update((k, o) for k, o in itertools.izip(instance.init_arguments, args))
                kwarg_sids = tuple((k, _calculate_sid(o, k))
                                   for k, o in sorted(kwargs.iteritems())
                                   if k not in instance.sid_ignore)
                instance.sid = (type(instance), kwarg_sids)
                ImmutableMeta.sids_created += 1
            except ValueError as e:
                instance.sid_failure = str(e)
        else:
            instance.sid_failure = 'disabled'

        instance._locked = True
        return instance

    __call__ = _call


class ImmutableInterface(BasicInterface):
    __metaclass__ = ImmutableMeta
    calculate_sid = True
    sid_ignore = ('name', 'caching')

    # Unlocking an immutable object will result in the deletion of its sid.
    # However, this will not delete the sids of objects referencing it.
    # You really should not unlock an object unless you really know what
    # you are doing. (One exception might be the modification of a newly
    # created copy of an immutable object.)
    def lock(self, doit=True):
        super(ImmutableInterface, self).lock(doit)
        if not self.locked and hasattr(self, 'sid'):
            del self.sid
            self.sid_failure = 'unlocked'

    def unlock(self):
        super(ImmutableInterface, self).unlock()
        if hasattr(self, 'sid'):
            del self.sid
            self.sid_failure = 'unlocked'
