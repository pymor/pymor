# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
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
    5. :meth:`BasicInterface.uid` provides a unique id for each instance. While
       `id(obj)` is only guaranteed to be unique among all living Python objects,
       :meth:`BasicInterface.uid` will be (almost) unique among all pyMOR objects
       that have ever existed, including previous runs of the application. This
       is achieved by building the id from a uuid4 which is newly created for
       each pyMOR run and a counter which is increased for any object that requests
       an uid.
    6. If not set by the user to another value, :attr:`BasicInterface.name` is
       generated from the class name and the :meth:`~BasicInterface.uid` of the
       instance.


:class:`ImmutableInterface` derives from :class:`BasicInterface` and adds the following
functionality:

    1. Using more metaclass magic, each instance which derives from
       :class:`ImmutableInterface` is locked after its `__init__` method has returned.
    2. A unique _`state id` for the instance can be calculated by calling
       :meth:`~ImmutableInterface.generate_sid` and is then stored as `sid` attribute.
       The state id is obtained by deterministically serializing the object's state
       and then computing a checksum of the resulting byte stream.
    3. :attr:`ImmutableInterface.sid_ignore` can be set to a set of attribute names
       which should be excluded from sid calculation.
    4. :meth:`ImmutableInterface.with_` can be used to create a copy of an instance with
       some changed attributes. E.g. ::

           obj.with_(a=x, b=y)

       creates a copy with the `a` and `b` attributes of `obj` set to `x` and `y`.
       (Note that in general `a` and `b` do not necessarily have to correspond to
       class attributes of `obj`; it is up to the implementor to interpret the
       provided arguments.) :attr:`ImmutableInterface.with_arguments` holds the
       set of allowed arguments.

       :class:`ImmutableInterface` provides a default implementation of `with_` which
       works by creating a new instance where the arguments of `with_` are passed
       through to `__init__`. The missing `__init__` arguments are taken from the
       corresponding instance attributes.
"""

import abc
try:
    from cPickle import dumps, HIGHEST_PROTOCOL
except ImportError:
    from pickle import dumps, HIGHEST_PROTOCOL
from copyreg import dispatch_table
import hashlib
import inspect
import itertools
import os
import time
from types import FunctionType, BuiltinFunctionType
import uuid
import sys

import numpy as np

from pymor.core import decorators, backports, logger
from pymor.core.exceptions import ConstError, SIDGenerationError

PY2 = sys.version_info.major == 2
DONT_COPY_DOCSTRINGS = int(os.environ.get('PYMOR_WITH_SPHINX', 0)) == 1
NoneType = type(None)

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

        I tell base classes when I derive a new class from them. I create a logger
        for each class I create. I add an `init_args` attribute to the class.
        """

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
        """I copy docstrings from base class methods to deriving classes.

        Copying of docstrings is disabled when the `PYMOR_WITH_SPHINX` environment
        variable is set to `1`.
        """
        for attr in ('_init_arguments', '_init_defaults'):
            if attr in classdict:
                raise ValueError(attr + ' is a reserved class attribute for subclasses of BasicInterface')

        for attr, item in classdict.items():
            if isinstance(item, FunctionType):
                # first copy/fixup docs
                item.__doc__ = decorators.fixup_docstring(item.__doc__)
                base_doc = None
                for base in bases:
                    base_func = getattr(base, item.__name__, None)
                    if not DONT_COPY_DOCSTRINGS:
                        if base_func:
                            base_doc = getattr(base_func, '__doc__', None)
                        if base_doc:
                            doc = decorators.fixup_docstring(getattr(item, '__doc__', ''))
                            if doc is not None:
                                base_doc = doc
                            item.__doc__ = base_doc

        c = abc.ABCMeta.__new__(cls, classname, bases, classdict)

        if PY2:
            try:
                args, varargs, keywords, defaults = inspect.getargspec(c.__init__)
                assert args[0] == 'self'
                c._init_arguments = tuple(args[1:])
            except TypeError:       # happens when no one declares an __init__ method and object is reached
                c._init_arguments = tuple()
        else:
            # getargspec is deprecated and does not work with keyword only args
            init_sig = inspect.signature(c.__init__)
            init_args = []
            for arg, description in init_sig.parameters.items():
                if arg == 'self':
                    continue
                if description.kind == description.POSITIONAL_ONLY:
                    raise TypeError('It should not be possible that {}.__init__ has POSITIONAL_ONLY arguments'.
                                    format(c))
                if description.kind in (description.POSITIONAL_OR_KEYWORD, description.KEYWORD_ONLY):
                    init_args.append(arg)
            c._init_arguments = tuple(init_args)

        return c


class BasicInterface(object, metaclass=UberMeta):
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
    """
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
        return n or type(self).__name__

    @name.setter
    def name(self, n):
        self._name = n

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


abstractmethod = abc.abstractmethod
abstractproperty = abc.abstractproperty

if PY2:
    # backport path for issue5867
    abstractclassmethod = backports.abstractclassmethod
    abstractstaticmethod = backports.abstractstaticmethod
else:
    abstractclassmethod = abc.abstractclassmethod
    abstractstaticmethod = abc.abstractstaticmethod


class ImmutableMeta(UberMeta):
    """Metaclass for :class:`ImmutableInterface`."""

    def __new__(cls, classname, bases, classdict):

        # Ensure that '_sid_contains_cycles' and 'sid' are contained in sid_ignore.
        # Otherwise sids of objects in reference cycles may depend on the order in which
        # generate_sid is called upon these objects.
        if 'sid_ignore' in classdict:
            classdict['sid_ignore'] = set(classdict['sid_ignore']) | {'_sid_contains_cycles', 'sid'}

        c = UberMeta.__new__(cls, classname, bases, classdict)

        c._implements_reduce = ('__reduce__' in classdict or '__reduce_ex__' in classdict
                                or any(getattr(base, '_implements_reduce', False) for base in bases))
        return c

    def _call(self, *args, **kwargs):
        instance = super(ImmutableMeta, self).__call__(*args, **kwargs)
        instance._locked = True
        return instance

    __call__ = _call


class ImmutableInterface(BasicInterface, metaclass=ImmutableMeta):
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

       Also note that mutable private attributes will cause false cache
       misses when these attributes enter |state id| calculation. If your
       implementation uses such attributes, you should therefore add their
       names to the :attr:`~ImmutableInterface.sid_ignore` set.

    Attributes
    ----------
    add_with_arguments
        Set of additional arguments for `with_`.
        (See :attr:`~ImmutableInterface.with_arguments`.)
    sid
        The objects state id. Only avilable after
        :meth:`~ImmutableInterface.generate_sid` has been called.
    sid_ignore
        Set of attributes not to include in sid calculation.
    with_arguments
        Set of allowed keyword arguments for `with_`. This is the
        union of the arguments names of `__init__` and the names
        specified via :attr:`~ImmutableInterface.add_with_arguments`.
    """
    sid_ignore = frozenset({'_locked', '_logger', '_name', '_uid', '_sid_contains_cycles', 'sid'})

    # Unlocking an immutable object will result in the deletion of its sid.
    # However, this will not delete the sids of objects referencing it.
    # You really should not unlock an object unless you really know what
    # you are doing. (One exception might be the modification of a newly
    # created copy of an immutable object.)
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

    def generate_sid(self, debug=False):
        """Generate a unique |state id| for the given object.

        The generated sid is stored in the object's `sid` attribute.

        Parameters
        ----------
        debug
            If `True`, produce some debug output.

        Returns
        -------
        The generated sid.
        """
        if hasattr(self, 'sid'):
            return self.sid
        else:
            return self._generate_sid(debug, tuple())

    def _generate_sid(self, debug, seen_immutables):
        sid_generator = _SIDGenerator()
        sid, has_cycles = sid_generator.generate(self, debug, seen_immutables)
        self.__dict__['sid'] = sid
        self.__dict__['_sid_contains_cycles'] = has_cycles
        return sid

    add_with_arguments = frozenset()

    @property
    def with_arguments(self):
        return set(self._init_arguments).union(getattr(self, 'add_with_arguments', []))

    def with_(self, **kwargs):
        """Returns a copy with changed attributes.

        The default implementation is to create a new class instance
        with the given keyword arguments as arguments for `__init__`.
        Missing arguments are obtained form instance attributes with the
        same name.

        Parameters
        ----------
        `**kwargs`
            Names of attributes to change with their new values. Each attribute name
            has to be contained in `with_arguments`.

        Returns
        -------
        Copy of `self` with changed attributes.
        """
        if not set(kwargs.keys()) <= self.with_arguments:
            raise ValueError('Changing "{}" using with() is not allowed in {} (only "{}")'.format(
                list(kwargs.keys()), self.__class__, self.with_arguments))

        # fill missing __init__ arguments using instance attributes of same name
        for arg in self._init_arguments:
            if arg not in kwargs:
                try:
                    kwargs[arg] = getattr(self, arg)
                except AttributeError:
                    raise ValueError('Cannot find missing __init__ argument "{}" for "{}" as attribute of "{}"'.format(
                        arg, self.__class__, self))

        c = type(self)(**kwargs)

        if self.logging_disabled:
            c.disable_logging()

        return c

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def generate_sid(obj, debug=False):
    """Generate a unique |state id| (sid) for the current state of the given object.

    Parameters
    ----------
    obj
        The object for which to compute the state sid.
    debug
        If `True`, produce some debug output.

    Returns
    -------
    The generated state id.
    """
    sid_generator = _SIDGenerator()
    return sid_generator.generate(obj, debug, tuple())[0]


# Helper classes for generate_sid

if PY2:
    import __builtin__
    STRING_TYPES = (str, __builtin__.unicode)
else:
    STRING_TYPES = (str, bytes)


class _SIDGenerator(object):

    def __init__(self):
        self.memo = {}
        self.logger = logger.getLogger('pymor.core.interfaces')

    def generate(self, obj, debug, seen_immutables):
        start = time.time()

        self.has_cycles = False
        self.seen_immutables = seen_immutables + (id(obj),)
        self.debug = debug
        state = self.deterministic_state(obj, first_obj=True)

        if debug:
            print('-' * 100)
            print('Deterministic state for ' + getattr(obj, 'name', str(obj)))
            print('-' * 100)
            print()
            import pprint
            pprint.pprint(state, indent=4)
            print()

        sid = hashlib.sha256(dumps(state, protocol=-1)).hexdigest()

        if debug:
            print('SID: {}, reference cycles: {}'.format(sid, self.has_cycles))
            print()
            print()

        name = getattr(obj, 'name', None)
        if name:
            self.logger.debug('{}: SID generation took {} seconds'.format(name, time.time() - start))
        else:
            self.logger.debug('SID generation took {} seconds'.format(time.time() - start))
        return sid, self.has_cycles

    def deterministic_state(self, obj, first_obj=False):
        v = self.memo.get(id(obj))
        if v:
            return(v)

        t = type(obj)
        if t in (NoneType, bool, int, float, FunctionType, BuiltinFunctionType, type):
            return obj

        self.memo[id(obj)] = _MemoKey(len(self.memo), obj)

        if t in STRING_TYPES:
            return obj

        if t is np.ndarray and t.dtype != object:
            return obj

        if t is tuple:
            return (tuple,) + tuple(self.deterministic_state(x) for x in obj)

        if t is list:
            return [self.deterministic_state(x) for x in obj]

        if t in (set, frozenset):
            return (t,) + tuple(self.deterministic_state(x) for x in sorted(obj))

        if t is dict:
            return (dict,) + tuple((k if type(k) is str else self.deterministic_state(k), self.deterministic_state(v))
                                   for k, v in sorted(obj.items()))

        if issubclass(t, ImmutableInterface):
            if hasattr(obj, 'sid') and not obj._sid_contains_cycles:
                return (t, obj.sid)

            if not first_obj:
                if id(obj) in self.seen_immutables:
                    raise _SIDGenerationRecursionError
                try:
                    obj._generate_sid(self.debug, self.seen_immutables)
                    return (t, obj.sid)
                except _SIDGenerationRecursionError:
                    self.has_cycles = True
                    self.logger.debug('{}: contains cycles of immutable objects, consider refactoring'.format(obj.name))

            if obj._implements_reduce:
                self.logger.debug('{}: __reduce__ is implemented, not using sid_ignore'.format(obj.name))
                return self.handle_reduce_value(obj, t, obj.__reduce_ex__(HIGHEST_PROTOCOL), first_obj)
            else:
                try:
                    state = obj.__getstate__()
                except AttributeError:
                    state = obj.__dict__
                state = {k: v for k, v in state.items() if k not in obj.sid_ignore}
                return self.deterministic_state(state) if first_obj else (t, self.deterministic_state(state))

        sid = getattr(obj, 'sid', None)
        if sid:
            return sid if first_obj else (t, sid)

        reduce = dispatch_table.get(t)
        if reduce:
            rv = reduce(obj)
        else:
            if issubclass(t, type):
                return obj

            reduce = getattr(obj, '__reduce_ex__', None)
            if reduce:
                rv = reduce(HIGHEST_PROTOCOL)
            else:
                reduce = getattr(obj, '__reduce__', None)
                if reduce:
                    rv = reduce()
                else:
                    raise SIDGenerationError('Cannot handle {} of type {}'.format(obj, t.__name__))

        return self.handle_reduce_value(obj, t, rv, first_obj)

    def handle_reduce_value(self, obj, t, rv, first_obj):
        if type(rv) is str:
            raise SIDGenerationError('__reduce__ methods returning a string are currently not handled '
                                     + '(object {} of type {})'.format(obj, t.__name__))

        if type(rv) is not tuple or not (2 <= len(rv) <= 5):
            raise SIDGenerationError('__reduce__ return value malformed '
                                     + '(object {} of type {})'.format(obj, t.__name__))

        rv = rv + (None,) * (5 - len(rv))
        func, args, state, listitems, dictitems = rv

        state = (func,
                 tuple(self.deterministic_state(x) for x in args),
                 self.deterministic_state(state),
                 self.deterministic_state(tuple(listitems)) if listitems is not None else None,
                 self.deterministic_state(sorted(dictitems)) if dictitems is not None else None)

        return state if first_obj else (t,) + state


class _MemoKey(object):
    def __init__(self, key, obj):
        self.key = key
        self.obj = obj

    def __repr__(self):
        return '_MemoKey({}, {})'.format(self.key, repr(self.obj))

    def __getstate__(self):
        return self.key


class _SIDGenerationRecursionError(Exception):
    pass
