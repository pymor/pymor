# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from functools import total_ordering

from pymor.core.interfaces import ImmutableInterface


@total_ordering
class BoundaryType(ImmutableInterface):
    """Represents a boundary type, i.e. Dirichlet, Neumann, etc.

    By defining a global registry of possible boundary types, we prevent hard
    to track down errors due to typos. Only boundary types that have been
    registered before using `register_type` can be instantiated.

    The boundary types which are registered by default are 'dirichlet',
    'neumann' and 'robin'.

    Parameters
    ----------
    `type_`
        Name of the boundary type as a string.

    Attributes
    ----------
    types
        Set of the names of registered boundary types.
    """

    types = {'dirichlet', 'neumann', 'robin'}

    @classmethod
    def register_type(cls, name):
        """Register a new |BoundaryType| with name `name`."""
        assert isinstance(name, str)
        cls.types.add(name)

    def __init__(self, type_):
        assert type_ in self.types, '{} is not a known boundary type. Use BoundaryType.register to add it'.format(type_)
        self.name = self.type = type_

    def __str__(self):
        return self.type

    def __repr__(self):
        return "BoundaryType('{}')".format(self.type)

    def __eq__(self, other):
        if isinstance(other, BoundaryType):
            return self.type == other.type
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, BoundaryType):
            return self.type != other.type
        else:
            return NotImplemented

    def __lt__(self, other):
        return self.type < other.type

    def __hash__(self):
        return hash(self.type)
