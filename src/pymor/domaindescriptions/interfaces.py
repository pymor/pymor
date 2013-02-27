from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from .boundarytypes import BoundaryType


class DomainDescriptionInterface(core.BasicInterface):
    '''Analytically describes a domain and its boundary (types).
    '''

    boundary_types = set()

    @property
    def has_dirichlet(self):
        return BoundaryType('dirichlet') in self.boundary_types

    @property
    def has_neumann(self):
        return BoundaryType('neumann') in self.boundary_types

    @property
    def has_only_dirichlet(self):
        return self.boundary_types == set((BoundaryType('dirichlet'),))

    @property
    def has_only_neumann(self):
        return self.boundary_types == set((BoundaryType('neumann'),))

    @property
    def has_only_dirichletneumann(self):
        return self.boundary_types <= set((BoundaryType('dirichlet'), BoundaryType('neumann')))

