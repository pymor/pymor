from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pymor.core as core
from pymor.common import BoundaryType

# Is one entity allowed to have mor than one boundary type?
class IBoundaryInfo(core.BasicInterface):
    '''Describes boundary types associated to a grid. For every boundary
    type and codimension a mask is provided, marking grid entities
    of the respective type and codimension by their global index.
    '''

    boundary_types = set()

    def mask(self, boundary_type, codim):
        '''retval[i] is True iff the codim-`codim` entity of global
        index `i` is associated to the boundary type `boundary_type`
        '''
        raise ValueError('Has no boundary_type "{}"'.format(boundary_type))

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

    def dirichlet_mask(self, codim):
        return self.mask(BoundaryType('dirichlet'), codim)

    def neumann_mask(self, codim):
        return self.mask(BoundaryType('neumann'), codim)

    def dirichlet_boundaries(self, codim):
        @core.cached
        def _dirichlet_boundaries(codim):
            return np.where(self.dirichlet_mask(codim))[0].astype('int32')
        return _dirichlet_boundaries(codim)

    def neumann_boundaries(self, codim):
        @core.cached
        def _neumann_boundaries(codim):
            return np.where(self.neumann_mask(codim))[0].astype('int32')
        return _neumann_boundaries(codim)
