# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides base classes from which most classes in pyMOR inherit.

The purpose of these classes is to provide some common functionality for
all objects in pyMOR. The most notable features provided by :class:`BasicInterface`
are the following:

    1. :class:`BasicInterface` sets class :class:`UberMeta` as metaclass
       which itself inherits from :class:`abc.ABCMeta`, so it is possible
       to define interface classes with abstract methods using the
       :func:`abstractmethod` decorator. There are also decorators for
       abstract class methods, static methods, and properties.
    2. Using metaclass magic, each *class* deriving from :class:`BasicInterface`
       comes with its own logger instance accessible through its `logger`
       attribute. The logger prefix is automatically set to the class name.
    3. Logging can be disabled and re-enabled for each *instance* using the
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
       (Note that in general `a` and `b` do not necessarily have to correspond to
       class attributes of `obj`; it is up to the implementor to interpret the
       provided arguments.) :attr:`BasicInterface.with_arguments` holds the
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
       `id(obj)` is only guaranteed to be unique among all living Python objects,
       :meth:`BasicInterface.uid` will be (almost) unique among all pyMOR objects
       that have ever existed, including previous runs of the application. This
       is achieved by building the id from a uuid4 which is newly created for
       each pyMOR run and a counter which is increased for any object that requests
       an uid.
    7. If not set by the user to another value, :attr:`BasicInterface.name` is
       generated from the class name and the :meth:`~BasicInterface.uid` of the
       instance.


:class:`ImmutableInterface` derives from :class:`BasicInterface` and adds the following
functionality:

    1. Using more metaclass magic, each instance which derives from
       :class:`ImmutableInterface` is locked after its `__init__` method has returned.
    2. If possible, a unique _`state id` for the instance is calculated and stored as
       `sid` attribute. If sid calculation fails, `sid_failure` is set to a string
       giving a reason for the failure.

       The basic idea behind state ids is that for an immutable object the result
       of any member function call is already pre-determined by the objects state id,
       the function's arguments and the state of pyMOR's global :mod:`~pymor.core.defaults`.

       The sid is constructed as a tuple containing:

           - the class of the instance
           - for each `__init__` argument its name and

               - its `sid` if it has one
               - its value if it is an instance of `NoneType`, `str`, `int`, `float` or `bool`
               - its value if it is a numpy array of short size

             For `tuple`, `list` or `dict` instances, the calculation is done by recursion.
             If none of these cases apply, sid calculation fails.
           - the state of all :mod:`~pymor.core.defaults` which have been set by the user

       .. warning::
          Default values defined in the function signature do not enter the sid
          calculation. Thus, if you change default values in your code, pyMOR will not
          be aware of these changes! As a consequence, you should always take
          care to clear your :mod:`~pymor.core.cache` if you change in-code default
          values.

       Note that a sid contains only object references to the sids of the provided `__init__`
       arguments. This structure is preserved by pickling resulting in relatively short
       string representations of the sid.
    3. :attr:`ImmutableInterface.sid_ignore` can be set to a tuple of `__init__`
       argument names, which should be excluded from sid calculation.
    4. sid generation (with all its overhead) can be disabled by setting
       :attr:`ImmutableInterface.calculate_sid` to `False`.
    5. sid generation can be disabled completely in pyMOR by calling
       :func:`disable_sid_generation`. It can be activated again by calling
       :func:`enable_sid_generation`.
"""

from __future__ import absolute_import, division, print_function
import abc
import inspect
import itertools
import os
import types
import uuid
from types import NoneType

try:
    import contracts
    HAVE_CONTRACTS = True
except ImportError:
    HAVE_CONTRACTS = False

import numpy as np

from pymor.core import decorators, backports, logger
from pymor.core.exceptions import ConstError
from pymor.core.defaults import defaults_sid
from pymor.tools.frozendict import FrozenDict

DONT_COPY_DOCSTRINGS = int(os.environ.get('PYMOR_COPY_DOCSTRINGS_DISABLE', 0)) == 1


class UID(object):
    '''Provides unique, quickly computed ids by combinding a session UUID4 with a counter.'''

    __slots__ = ['uid']

    prefix = '{}_'.format(uuid.uuid4())
    counter = [0]

    def __init__(self):
        self.uid = self.prefix + str(self.counter[0])
        self.counter[0] += 1

    def __getstate__(self):
        return 1

    def __setstate__(self, v):
        self.uid = self.prefix + str(self.counter[0])
        self.counter[0] += 1


class UberMeta(abc.ABCMeta):

    def __init__(cls, name, bases, namespace):
        """Metaclass of :class:`BasicInterface`.

        I tell base classes when I derive a new class from them. I publish
        a new contract type for each new class I create. I create a logger
        for each class I create. I add an `init_args` attribute to the class.
        """
        # monkey a new contract into the decorator module so checking for that type at runtime can work
        if HAVE_CONTRACTS:
            dname = (cls.__module__ + '.' + name).replace('__main__.', 'main.').replace('.', '_')
            if dname not in decorators.__dict__:
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
        cls._logger = logger.getLogger('{}.{}'.format(cls.__module__.replace('__main__', 'pymor'), name))
        abc.ABCMeta.__init__(cls, name, bases, namespace)

    def __new__(cls, classname, bases, classdict):
        """I copy contract decorations and docstrings from base class methods to deriving classes.
        I also forward "abstract{class|static}method" decorations in the base class to "{class|static}method"
        decorations in the new subclass.

        Copying of docstrings can be prevented by setting the `PYMOR_COPY_DOCSTRINGS_DISABLE` environment
        variable to `1`.
        """
        for attr in ('_init_arguments', '_init_defaults'):
            if attr in classdict:
                raise ValueError(attr + ' is a reserved class attribute for subclasses of BasicInterface')

        for attr, item in classdict.items():
            if isinstance(item, types.FunctionType):
                # first copy/fixup docs
                item.__doc__ = decorators.fixup_docstring(item.__doc__)
                base_doc = None
                contract_kwargs = dict()
                for base in bases:
                    base_func = getattr(base, item.__name__, None)
                    if not DONT_COPY_DOCSTRINGS:
                        has_contract = False
                        if base_func:
                            base_doc = getattr(base_func, '__doc__', None)
                            if HAVE_CONTRACTS:
                                has_contract = getattr(base_func, 'decorated', None) == 'contract'
                                contract_kwargs = getattr(base_func, 'contract_kwargs', contract_kwargs)
                        if base_doc:
                            doc = decorators.fixup_docstring(getattr(item, '__doc__', ''))
                            if HAVE_CONTRACTS:
                                has_base_contract_docs = decorators.contains_contract(base_doc)
                                has_contract_docs = decorators.contains_contract(doc)
                                if has_base_contract_docs and not has_contract_docs:
                                    base_doc += doc
                                elif not has_base_contract_docs and doc is not None:
                                    base_doc = doc
                            elif doc is not None:
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
        try:
            args, varargs, keywords, defaults = inspect.getargspec(c.__init__)
            assert args[0] == 'self'
            c._init_arguments = tuple(args[1:])
            if defaults:
                c._init_defaults = dict(zip(args[-len(defaults):], defaults))
            else:
                c._init_defaults = dict()
        except TypeError:       # happens when no one declares an __init__ method and object is reached
            c._init_arguments = tuple()
            c._init_defaults = dict()
        return c


class BasicInterface(object):
    """Base class for most classes in pyMOR.

    Attributes
    ----------
    locked
        `True` if the instance is made immutable using `lock`.
    logger
        A per-class instance of :class:`logging.Logger` with the class
        name as prefix.
    logging_disabled
        `True` if logging has been disabled.
    name
        The name of the instance. If not set by the user, the name is
        generated from the class name and the `uid` of the instance.
    uid
        A unique id for each instance. The uid is obtained by using
        :class:`UIDProvider` and should be unique for all pyMOR objects
        ever created.
    with_arguments
        Set of allowed keyword arguments for `with_`.
    """

    __metaclass__ = UberMeta
    _locked = False

    def __setattr__(self, key, value):
        """depending on _locked state I delegate the setattr call to object or
        raise an Exception
        """
        if not self._locked or key[0] == '_':
            return object.__setattr__(self, key, value)
        else:
            raise ConstError('Changing "%s" is not allowed in locked "%s"' % (key, self.__class__))

    @property
    def locked(self):
        return self._locked

    def lock(self, doit=True):
        """Make the instance immutable.

        Trying to change an attribute after locking raises a `ConstError`.
        Private attributes (of the form `_attribute`) are exempted from
        this rule.
        """
        object.__setattr__(self, '_locked', doit)

    def unlock(self):
        """Make the instance mutable again, after it has been locked using `lock`."""
        object.__setattr__(self, '_locked', False)

    @property
    def name(self):
        n = getattr(self, '_name', None)
        return n or '{}_{}'.format(type(self).__name__, self.uid)

    @name.setter
    def name(self, n):
        self._name = n

    @property
    def with_arguments(self):
        init_arguments = self._init_arguments
        for arg in init_arguments:
            if not hasattr(self, arg):
                self._with_arguments_error = "Instance does not have attribute for __init__ argument '{}'".format(arg)
                return set()
        return set(init_arguments)

    def with_(self, **kwargs):
        """Returns a copy with changed attributes.

        The default implementation is to call `_with_via_init(**kwargs)`.

        Parameters
        ----------
        `**kwargs`
            Names of attributes to change with their new values. Each attribute name
            has to be contained in `with_arguments`.

        Returns
        -------
        Copy of `self` with changed attributes.
        """
        with_arguments = self.with_arguments      # ensure that property is called first
        if hasattr(self, '_with_arguments_error'):
            raise ConstError('Using with_ is not possible because of the following Error: '
                             + self._with_arguments_error)
        if not set(kwargs.keys()) <= with_arguments:
            raise ConstError('Changing "{}" using with() is not allowed in {} (only "{}")'.format(
                kwargs.keys(), self.__class__, self.with_arguments))
        return self._with_via_init(kwargs)

    def _with_via_init(self, kwargs, new_class=None):
        """Default implementation for with_ by calling __init__.

        Parameters which are missing in `kwargs` are taken from the dictionary of the
        instance. If `new_class` is provided, the copy is created as an instance of
        `new_class`.
        """
        my_type = type(self) if new_class is None else new_class
        init_args = kwargs
        for arg in my_type._init_arguments:
            if arg not in init_args:
                init_args[arg] = getattr(self, arg)
        c = my_type(**init_args)
        if self.logging_disabled:
            c.disable_logging()
        return c

    _added_attributes = None

    def add_attributes(self, **kwargs):
        """Add attributes to a locked instance."""
        assert not any(hasattr(self, k) for k in kwargs)
        self.__dict__.update(kwargs)
        if self._added_attributes is None:
            self._added_attributes = kwargs.keys()
        else:
            self._added_attributes.extend(kwargs.keys())

    @property
    def logging_disabled(self):
        return self._logger is logger.dummy_logger

    @property
    def logger(self):
        return self._logger

    def disable_logging(self, doit=True):
        """Disable logging output for this instance."""
        if doit:
            self._logger = logger.dummy_logger
        else:
            del self._logger

    def enable_logging(self, doit=True):
        """Enable logging output for this instance."""
        self.disable_logging(not doit)

    @classmethod
    def implementors(cls, descend=False):
        """I return a, potentially empty, list of my subclass-objects.
        If descend is True I traverse my entire subclass hierarchy and return a flattened list.
        """
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
        """For convenience I return a list of my implementor names instead of class objects"""
        return [c.__name__ for c in cls.implementors(descend)]

    @classmethod
    def has_interface_name(cls):
        """`True` if the class name ends with `Interface`. Used for introspection."""
        name = cls.__name__
        return name.endswith('Interface')

    _uid = None

    @property
    def uid(self):
        if self._uid is None:
            self._uid = UID()
        return self._uid.uid


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
    """I mark my wrapped function with an additional __isabstractclassmethod__ member,
    where my abstractclassmethod_base sets __isabstractmethod__ = True.
    """

    def __init__(self, callable_method):
        callable_method.__isabstractclassmethod__ = True
        super(abstractclassmethod, self).__init__(callable_method)


class abstractstaticmethod(abstractstaticmethod_base):
    """I mark my wrapped function with an additional __isabstractstaticmethod__ member,
    where my abstractclassmethod_base sets __isabstractmethod__ = True.
    """

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
        elif t_obj is dict or t_obj is FrozenDict:
            return tuple((k, _calculate_sid(v, '{}[{}]'.format(name, k))) for k, v in sorted(obj.iteritems()))
        elif t_obj in (NoneType, str, int, float, bool):
            return obj
        elif t_obj is np.ndarray:
            if obj.size < 64:
                return _calculate_sid(obj.tolist(), name)
            else:
                raise ValueError('sid calculation faild at large numpy array {}'.format(name))
        else:
            raise ValueError('sid calculation failed at {}={}'.format(name, type(obj)))


def inject_sid(obj, context, *args):
    """Add a state id sid to an object.

    The purpose of this methods is to inject state ids into objects which do not
    derive from :class:`ImmutableInterface`. If `obj` is an instance of
    :class:`BasicInterface`, it is locked, if it is an :class:`numpy.ndarray`,
    its `writable` flag is set to `False`.

    It is the callers responsibility to ensure that the given parameters uniquely
    describe the state of `obj`, and that `obj` does not change its state after
    the call of `inject_sid`. For an example see
    :class:`pymor.analyticalproblems.EllipticProblem`.

    Parameters
    ----------
    obj
        The object which shall obtain a sid.
    context
        A hashable, picklable, immutable object, describing the context in
        which `obj` was created.
    `*args`
        List of parameters which in the given context led to the creation of
        `obj`.
    """
    try:
        sid = tuple((context, tuple(_calculate_sid(o, i) for i, o in enumerate(args)), defaults_sid()))
        obj.sid = sid
        ImmutableMeta.sids_created += 1
    except ValueError as e:
        obj.sid_failure = str(e)

    if isinstance(obj, BasicInterface):
        obj.lock()
    elif isinstance(obj, np.ndarray):
        obj.flags.writable = False


def disable_sid_generation():
    """Globally disable the generation of state ids."""
    if hasattr(ImmutableMeta, '__call__'):
        del ImmutableMeta.__call__


def enable_sid_generation():
    """Globally enable the generation of state ids."""
    ImmutableMeta.__call__ = ImmutableMeta._call


class ImmutableMeta(UberMeta):
    """Metaclass for :class:`ImmutableInterface`."""

    sids_created = 0
    init_arguments_never_warn = ('name', 'cache_region')

    def __new__(cls, classname, bases, classdict):
        c = UberMeta.__new__(cls, classname, bases, classdict)
        init_arguments = c._init_arguments
        try:
            for a in c.sid_ignore:
                if a not in init_arguments and a not in ImmutableMeta.init_arguments_never_warn:
                    raise ValueError(a)
        except ValueError as e:
            # The _logger attribute of our new class has not been initialized yet, so create
            # our own logger.
            l = logger.getLogger('{}.{}'.format(c.__module__.replace('__main__', 'pymor'), classname))
            l.warn('sid_ignore contains "{}" which is not an __init__ argument!'.format(e))
        return c

    def _call(self, *args, **kwargs):
        instance = super(ImmutableMeta, self).__call__(*args, **kwargs)
        if instance.calculate_sid:
            try:
                arguments = instance._init_defaults.copy()
                arguments.update(kwargs)
                arguments.update((k, o) for k, o in itertools.izip(instance._init_arguments, args))
                arguments_sids = tuple((k, _calculate_sid(o, k))
                                       for k, o in sorted(arguments.iteritems())
                                       if k not in instance.sid_ignore)
                instance.sid = (type(instance), arguments_sids, defaults_sid())
                ImmutableMeta.sids_created += 1
            except ValueError as e:
                instance.sid_failure = str(e)
        else:
            instance.sid_failure = 'disabled'

        instance._locked = True
        return instance

    __call__ = _call


class ImmutableInterface(BasicInterface):
    """Base class for immutable objects in pyMOR.

    Instances of `ImmutableInterface` are immutable in the sense that
    they are :meth:`locked <BasicInterface.lock>` after `__init__` returns.

    .. _ImmutableInterfaceWarning:
    .. warning::
       For instances of `ImmutableInterface`, the following should always
       be true ::

           The result of any member function call is determined by the
           function's arguments together with the state id of the
           corresponding instance and the current state of pyMOR's
           global defaults.

       While, in principle, you are allowed to modify private members after
       instance initialization, this should never affect the outcome of
       future method calls. In particular, if you update any internal state
       after initialization, you have to ensure that this state is not affecteed
       by possible changes of the global :mod:`~pymor.core.defaults`.

    Attributes
    ----------
    calculate_sid
        If `True`, a unique id describing the state of the instance is
        calculated after __init__ returns, based on the states of the
        provided arguments. For further details see
        :mod:`pymor.core.interfaces`.
    sid
        The objects state id. If sid generation is disabled or fails,
        this attribute is not set.
    sid_failure
        If sid generation fails, a string describing the reason for
        the failure.
    sid_ignore
        Tuple of `__init__` arguments not to include in sid calculation.
        The default it `{'name', 'cache_region'}`
    """
    __metaclass__ = ImmutableMeta
    calculate_sid = True
    sid_ignore = frozenset({'name', 'cache_region'})

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
        """Make the instance mutable.

        .. warning::
            Unlocking an instance of :class:`ImmutableInterface` will result in the
            deletion of its sid. However, this will not delete the sids of
            objects referencing it. You really should not unlock an object
            unless you really know what you are doing. (One exception might
            be the modification of a newly created copy of an immutable object.)
        """
        super(ImmutableInterface, self).unlock()
        if hasattr(self, 'sid'):
            del self.sid
            self.sid_failure = 'unlocked'
