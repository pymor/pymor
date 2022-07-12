# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""This module provides base classes from which most classes in pyMOR inherit.

The purpose of these classes is to provide some common functionality for
all objects in pyMOR. The most notable features provided by :class:`BasicObject`
are the following:

    1. :class:`BasicObject` sets class :class:`UberMeta` as metaclass
       which itself inherits from :class:`abc.ABCMeta`. Thus it is possible
       to define interface classes with abstract methods using the
       :func:`abstractmethod` decorator. There are also decorators for
       abstract class methods, static methods, and properties.
    2. Using metaclass magic, each *class* deriving from :class:`BasicObject`
       comes with its own :mod:`~pymor.core.logger` instance accessible through its `logger`
       attribute. The logger prefix is automatically set to the class name.
    3. Logging can be disabled and re-enabled for each *instance* using the
       :meth:`BasicObject.disable_logging` and :meth:`BasicObject.enable_logging`
       methods.
    4. :meth:`BasicObject.uid` provides a unique id for each instance. While
       `id(obj)` is only guaranteed to be unique among all living Python objects,
       :meth:`BasicObject.uid` will be (almost) unique among all pyMOR objects
       that have ever existed, including previous runs of the application. This
       is achieved by building the id from a uuid4 which is newly created for
       each pyMOR run and a counter which is increased for any object that requests
       an uid.
    5. If not set by the user to another value, :attr:`BasicObject.name` is
       set to the name of the object's class.


:class:`ImmutableObject` derives from :class:`BasicObject` and adds the following
functionality:

    1. Using more metaclass magic, each instance which derives from
       :class:`ImmutableObject` is locked after its `__init__` method has returned.
       Each attempt to change one of its attributes raises an exception. Private
       attributes (of the form `_name`) are exempted from this rule.
    2. :meth:`ImmutableObject.with_` can be used to create a copy of an instance with
       some changed attributes. E.g. ::

           obj.with_(a=x, b=y)

       creates a copy with the `a` and `b` attributes of `obj` set to `x` and `y`.
       `with_` is implemented by creating a new instance, passing the arguments of
       `with_` to `__init__`. The missing `__init__` arguments are taken from instance
       attributes of the same name.
"""

import abc
from functools import wraps
import inspect
import os
from types import FunctionType
import uuid

from pymor.core import logger
from pymor.core.exceptions import ConstError
from pymor.tools.formatrepr import format_repr, _format_generic

DONT_COPY_DOCSTRINGS = int(os.environ.get('PYMOR_WITH_SPHINX', 0)) == 1
NoneType = type(None)


class UID:
    """Provides unique, quickly computed ids by combining a session UUID4 with a counter."""

    __slots__ = ['uid']

    prefix = f'{uuid.uuid4()}_'
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
        """Metaclass of :class:`BasicObject`.

        I create a logger for each class I create.
        I add an `init_arguments` attribute to the class.
        """
        cls._logger = logger.getLogger(f'{cls.__module__.replace("__main__", "pymor")}.{name}')
        abc.ABCMeta.__init__(cls, name, bases, namespace)

    def __new__(cls, classname, bases, classdict):
        """I copy docstrings from base class methods to deriving classes.

        Copying of docstrings is disabled when the `PYMOR_WITH_SPHINX` environment
        variable is set to `1`.
        """
        for attr in ('_init_arguments', '_init_defaults'):
            if attr in classdict:
                raise ValueError(attr + ' is a reserved class attribute for subclasses of BasicObject')

        for attr, item in classdict.items():
            if isinstance(item, FunctionType):
                # first copy docs
                base_doc = None
                for base in bases:
                    base_func = getattr(base, item.__name__, None)
                    if not DONT_COPY_DOCSTRINGS:
                        if base_func:
                            base_doc = getattr(base_func, '__doc__', None)
                        if base_doc:
                            doc = getattr(item, '__doc__', '')
                            if doc is not None:
                                base_doc = doc
                            item.__doc__ = base_doc

        def __auto_init(self, locals_):
            """Automatically assign __init__ arguments.

            This method is used in __init__ to automatically assign __init__ arguments to equally
            named object attributes. The values are provided by the `locals_` dict. Usually,
            `__auto_init` is called as::

                self.__auto_init(locals())

            where `locals()` returns a dictionary of all local variables in the current scope.
            Only attributes which have not already been set by the user are initialized by
            `__auto_init`.
            """
            for arg in c._init_arguments:
                if arg not in self.__dict__:
                    setattr(self, arg, locals_[arg])

        auto_init_name = f'_{classname}__auto_init'
        classdict[auto_init_name] = __auto_init
        c = abc.ABCMeta.__new__(cls, classname, bases, classdict)
        # by updating the qualified name we make filtering in sphinx possible
        getattr(c, auto_init_name).__qualname__ = auto_init_name

        init_sig = inspect.signature(c.__init__)
        init_args = []
        has_args, has_kwargs = False, False
        for arg, description in init_sig.parameters.items():
            if arg == 'self':
                continue
            if description.kind in (description.POSITIONAL_OR_KEYWORD, description.POSITIONAL_ONLY,
                                    description.KEYWORD_ONLY):
                init_args.append(arg)
            elif description.kind == description.VAR_POSITIONAL:
                has_args = True
            elif description.kind == description.VAR_KEYWORD:
                has_kwargs = True
            else:
                raise NotImplementedError(f'Unknown argument type {description.kind}')
        c._init_arguments, c._init_has_args, c._init_has_kwargs = tuple(init_args), has_args, has_kwargs

        return c


class BasicObject(metaclass=UberMeta):
    """Base class for most classes in pyMOR.

    Attributes
    ----------
    logger
        A per-class instance of :class:`logging.Logger` with the class
        name as prefix.
    logging_disabled
        `True` if logging has been disabled.
    name
        The name of the instance. If not set by the user, the name is
        set to the class name.
    uid
        A unique id for each instance. The uid is obtained by using
        :class:`UID` and is unique for all pyMOR objects ever created.
    """

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

    _uid = None

    @property
    def uid(self):
        if self._uid is None:
            self._uid = UID()
        return self._uid.uid

    def _format_repr(self, max_width, verbosity, override={}):
        if verbosity < 3 and self.name == type(self).__name__ and 'name' not in override:
            override = dict(override, name=None)
        return _format_generic(self, max_width, verbosity, override=override)

    def __repr__(self):
        return format_repr(self)


abstractmethod = abc.abstractmethod
abstractproperty = abc.abstractproperty
abstractclassmethod = abc.abstractclassmethod
abstractstaticmethod = abc.abstractstaticmethod


class classinstancemethod:

    def __init__(self, cls_meth):
        self.cls_meth = cls_meth

    def __get__(self, instance, cls):
        if cls is None:
            return self
        if instance is None:
            @wraps(self.cls_meth)
            def the_class_method(*args, **kwargs):
                return self.cls_meth(cls, *args, **kwargs)
            the_class_method.autoapi_skip = True
            return the_class_method
        else:
            @wraps(self.inst_meth)
            def the_instance_method(*args, **kwargs):
                return self.inst_meth(instance, *args, **kwargs)
            return the_instance_method

    def instancemethod(self, inst_meth):
        inst_meth.__doc__ = inst_meth.__doc__ or self.cls_meth.__doc__
        self.inst_meth = inst_meth
        return self


class ImmutableMeta(UberMeta):
    """Metaclass for :class:`ImmutableObject`."""

    def __new__(cls, classname, bases, classdict):

        c = UberMeta.__new__(cls, classname, bases, classdict)

        c._implements_reduce = ('__reduce__' in classdict
                                or '__reduce_ex__' in classdict
                                or any(getattr(base, '_implements_reduce', False)
                                       for base in bases))

        # set __signature__ attribute on newly created class c to ensure that
        # inspect.signature(c) returns the signature of its __init__ arguments and not
        # the signature of ImmutableMeta.__call__
        sig = inspect.signature(c.__init__)
        c.__signature__ = sig.replace(parameters=tuple(sig.parameters.values())[1:])
        return c

    def _call(self, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        assert all(hasattr(instance, arg) for arg in instance._init_arguments), \
            (f'__init__ arguments {[arg for arg in instance._init_arguments if not hasattr(instance,arg)]} '
             f'of class {self.__name__} not available as instance attributes\n'
             f'(all __init__ args need to be attributes for with_ to work).')
        instance._locked = True
        return instance

    __call__ = _call


class ImmutableObject(BasicObject, metaclass=ImmutableMeta):
    """Base class for immutable objects in pyMOR.

    Instances of `ImmutableObject` are immutable in the sense that
    after execution of `__init__`, any modification of a non-private
    attribute will raise an exception.

    .. _ImmutableObjectWarning:
    .. warning::
           For instances of `ImmutableObject`,
           the result of member function calls should be completely
           determined by the function's arguments together with the
           object's `__init__` arguments and the current state of pyMOR's
           global |defaults|.

    While, in principle, you are allowed to modify private members after
    instance initialization, this should never affect the outcome of
    future method calls. In particular, if you update any internal state
    after initialization, you have to ensure that this state is not affected
    by possible changes of the global :mod:`~pymor.core.defaults`.
    """

    _locked = False

    # we need to define __init__, otherwise the Python 2 signature hack will fail
    def __init__(self):
        pass

    def __setattr__(self, key, value):
        """Depending on _locked state delegate the setattr call to object or raise an Exception."""
        if not self._locked or key[0] == '_':
            return object.__setattr__(self, key, value)
        else:
            raise ConstError(f'Changing "{key}" is not allowed in locked "{self.__class__}"')

    def with_(self, new_type=None, **kwargs):
        """Returns a copy with changed attributes.

        A a new class instance is created with the given keyword arguments as
        arguments for `__init__`.  Missing arguments are obtained form instance
        attributes with the
        same name.

        Parameters
        ----------
        new_type
            If not None, return an instance of this class (instead of `type(self)`).
        `**kwargs`
            Names of attributes to change with their new values. Each attribute name
            has to be an argument to `__init__`.

        Returns
        -------
        Copy of `self` with changed attributes.
        """
        # fill missing __init__ arguments using instance attributes of same name
        for arg in (self._init_arguments if new_type is None else new_type._init_arguments):
            if arg not in kwargs:
                try:
                    kwargs[arg] = getattr(self, arg)
                except AttributeError as e:
                    raise ValueError(f"Cannot find missing __init__ argument '{arg}' for '{self.__class__}' "
                                     f"as attribute of '{self}'") from e

        c = (type(self) if new_type is None else new_type)(**kwargs)

        if self.logging_disabled:
            c.disable_logging()

        return c

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
