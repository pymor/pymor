# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function


class Named(object):
    '''Mixin for objects with a name.

    Attributes
    ----------
    name
        Property returning the object's name. If no name has
        been set, a random name is created from the class name
        and a uuid4.
    '''

    __name = None

    @property
    def name(self):
        if self.__name is None:
            import uuid
            self.__name = '{}_{}'.format(self.__class__.__name__, uuid.uuid4())
        return self.__name

    @name.setter
    def name(self, n):
        self.__name = n
