from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core


class BoundaryType(core.BasicInterface):
    '''Represents the different boundary types there are. By defining a global register
    of possible boundary types, we prevent hard to track down errors due to typos.
    '''

    types = set(('dirichlet', 'neumann', 'robin'))

    @classmethod
    def register_type(cls, name):
        assert isinstance(name, str)
        cls.types.add(name)

    def __init__(self, name):
        assert name in self.types, '{} is not a known boundary type. Use BoundaryType.register to add it'.format(name)
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "BoundaryType('{}')".format(self.name)

    def __eq__(self, other):
        if isinstance(other, BoundaryType):
            return self.name == other.name
        #elif isinstance(other, str):       better not ...
        #    return self.name == other
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.name)
