# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module contains pure python backports of library features from python >= 3 to 2.7
"""
from __future__ import absolute_import, division, print_function


class abstractclassmethod(classmethod):
    """
    A decorator indicating abstract classmethods.

    Similar to abstractmethod.

    Usage::

        class C(metaclass=ABCMeta):
            @abstractclassmethod
            def my_abstract_classmethod(cls, ...):
                ...

    'abstractclassmethod' is deprecated. Use 'classmethod' with
    'abstractmethod' instead.
    """

    __isabstractmethod__ = True

    def __init__(self, callable_method):
        callable_method.__isabstractmethod__ = True
        super(abstractclassmethod, self).__init__(callable_method)


class abstractstaticmethod(staticmethod):
    """
    A decorator indicating abstract staticmethods.

    Similar to abstractmethod.

    Usage::

        class C(metaclass=ABCMeta):
            @abstractstaticmethod
            def my_abstract_staticmethod(...):
                ...

    'abstractstaticmethod' is deprecated. Use 'staticmethod' with
    'abstractmethod' instead.
    """

    __isabstractmethod__ = True

    def __init__(self, callable_method):
        callable_method.__isabstractmethod__ = True
        super(abstractstaticmethod, self).__init__(callable_method)
