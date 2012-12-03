from __future__ import absolute_import, division, print_function, unicode_literals

# For python3.2 and greater, we user functools.lru_cache for caching. If our python
# version is to old, we import the same decorator from the third-party functools32
# package
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

import numpy as np

import pymor.core as core


class IBoundaryInfo(core.BasicInterface):
    '''Describes boundary conditions associated to a grid. For every boundary
    condition type and codimension a mask is provided, marking grid entities
    of the respective type and codimension by their global index. Additionaly,
    for every condition type a data function is provided, mapping global
    coordinates to boundary data values.
    '''

    condition_types = set()

    def mask(self, condition_type, codim):
        '''retval[i] is True iff the codim-`codim` entity of global
        index `i` is associated to the boundary condition of type
        `condition_type`
        '''
        raise ValueError('Has no condition_type "{}"'.format(condition_type))

    def data(self, condition_type):
        '''returns a vectorized function mapping global coordinates to
        boundary data for the boundary condition of type 'condition_type'
        '''
        raise ValueError('Has no condition_type "{}"'.format(condition_type))

    @property
    def has_dirichlet(self):
        return 'dirichlet' in self.condition_types

    @property
    def has_neumann(self):
        return 'neumann' in self.condition_types

    @property
    def has_only_dirichlet(self):
        return self.condition_types == set(('dirichlet',))

    @property
    def has_only_neumann(self):
        return self.condition_types == set(('neumann',))

    @property
    def has_only_dirichletneumann(self):
        return self.condition_types <= set(('dirichlet', 'neumann'))

    def dirichlet_mask(self, codim):
        return self.mask('dirichlet', codim)

    def neumann_mask(self, codim):
        return self.mask('neumann', codim)

    def dirichlet_boundaries(self, codim):
        @lru_cache(maxsize=None)
        def _dirichlet_boundaries(codim):
            return np.where(self.dirichlet_mask(codim))[0].astype('int32')
        return _dirichlet_boundaries(codim)

    def neumann_boundaries(self, codim):
        @lru_cache(maxsize=None)
        def _neumann_boundaries(codim):
            return np.where(self.neumann_mask(codim))[0].astype('int32')
        return _neumann_boundaries(codim)

    def dirichlet_data(self, points):
        return self.data('dirichlet')(points)

    def neumann_data(self, points):
        return self.data('neumann')(points)
