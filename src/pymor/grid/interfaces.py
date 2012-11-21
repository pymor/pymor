from __future__ import absolute_import, division, print_function, unicode_literals

import pymor.core as core
from .defaultimpl import *


class IConformalTopologicalGrid(IConformalTopologicalGridDefaultImplementation, core.BasicInterface):
    '''Base interface for all grids. This is an incomplete prepreliminary version.
    Until now, only the toplogy part of the interface is specified in here.

    All results in the default implementations are cached for the next evaluation
    using functools.lru_cache. Note that the current implementation is very slow
    and should be moved to C code.
    For g = pymor.grid.rect.Rect((1000, 1000)) it takes around 4.5
    minutes on modern hardware to calculate g.neighbours(0, 1, 2).
    (The call involves calculating g.subentites(0, 2) and g.superentities(2, 1)
    at the same time.)
    '''

    @core.interfaces.abstractmethod
    def size(self, codim):
        '''size(codim) is the number of entities in the grid of codimension codim'''
        pass

    @core.interfaces.abstractmethod
    def subentities(self, codim, subentity_codim=None):
        '''retval[e,s] is the global index of the s-th codim-"subentity_codim"
        subentity of the codim-"codim" entity with global index e.

        If subentity_codim=None, it is set to codim+1.

        Only subentities(codim, None) has to be implemented, by default,
        subentities(codim, subentity_codim) is computed by calculating the
        transitive closure of subentities(codim, None)
        '''
        return self._subentities(codim, subentity_codim)

    def superentities(self, codim, superentity_codim=None):
        '''retval[e,s] is the global index of the s-th codim-"superentity_codim"
        superentity of the codim-"codim" entity with global index e.

        If superentity_codim == None, it is set to codim-1.

        The default implementation is to compute the result from subentities()
        '''
        return self._superentities(codim, superentity_codim)

    def superentity_indices(self, codim, superentity_codim=None):
        return self._superentity_indices(codim, superentity_codim)

    def neighbours(self, codim, neighbour_codim, intersection_codim=None):
        '''retval[e,s] is the global index of the n-th codim-"neighbour_codim"
        entitiy of the codim-"codim" entity with global index e that shares
        with it an intersection of codimension "intersection_codim".

        If intersection_codim == None,
            it is set to codim if codim == neighbour_codim
            otherwise it is set to min(codim, neighbour_codim).

        The default implementation is to compute the result from subentities()
        and superentities().
        '''
        return self._neighbours(codim, neighbour_codim, intersection_codim)


class ISimpleReferenceElement(ISimpleReferenceElementDefaultImplementation, core.BasicInterface):
    '''Defines a reference element with the property that each of its subentities is
    of the same type. I.e. a three-dimensional reference element cannot have triangles
    and rectangles as faces at the same time
    '''

    dim = None
    volume = None

    @core.interfaces.abstractmethod
    def size(self, codim):
        'Number of subentites of codimension "codim"'

    @core.interfaces.abstractmethod
    def subentities(self, codim, subentity_codim):
        '''subentities(c,sc)[i,j] is - with respect to the indexing inside the
        reference element  - the index of the j-th "subentity_codim"-codim
        subentity of the i-th "codim"-codim subentity of the reference element
        '''
        pass

    @core.interfaces.abstractmethod
    def subentity_embedding(self, subentity_codim):
        '''returns a tuple (A, B) which defines the embedding of the "subentity_codim"-
        subentity with index "index" into the reference element.
        for subsubentites, the embedding is by default given via its embedding into its
        lowest index superentity
        '''
        return self._subentity_embedding(subentity_codim)

    @core.interfaces.abstractmethod
    def sub_reference_element(self, codim):
        return self._sub_reference_element(codim)

    def __call__(self, codim):
        return self.sub_reference_element(codim)

    @core.interfaces.abstractmethod
    def unit_outer_normals(self):
        pass

    @core.interfaces.abstractmethod
    def center(self):
        pass

    @core.interfaces.abstractmethod
    def mapped_diameter(self, A):
        pass

    @core.interfaces.abstractmethod
    def quadrature(order=None, npoints=None, quadrature_type='default'):
        '''returns tuple (P, W) where P is an array of quadrature points with corresponding weights W for
        the given integration order "order" or with "npoints" integration points
        '''
        pass


class ISimpleAffineGrid(ISimpleAffineGridDefaultImplementation, IConformalTopologicalGrid):

    dim = None
    dim_outer = None

    @core.interfaces.abstractmethod
    def reference_element(self, codim):
        pass

    @core.interfaces.abstractmethod
    def subentities(self, codim, subentity_codim=None):
        '''retval[e,s] is the global index of the s-th codim-"subentity_codim"
        subentity of the codim-"codim" entity with global index e.

        If subentity_codim=None, it is set to codim+1.

        If codim > 0, we calculate the subentites of e by default as follows:
        - find the codim-0 parent entity e_0 with minimal global index
        - lookup the local indicies of the subentites of e inside e_0
          using the reference element
        - map these local indicies to global indicies using
          subentities(0, subentity_codim)
        This procedures assures that subentities(codim, subentity_codim)[i]
        has the right order w.r.t. the embedding determined by e_0, which
        is also the embedding return by embeddings(codim)
        '''
        return self._subentities(codim, subentity_codim)

    @core.interfaces.abstractmethod
    def embeddings(self, codim):
        return self._embeddings(codim)

    def jacobian_inverse_transposed(self, codim):
        return self._jacobian_inverse_transposed(codim)

    def integration_element(self, codim):
        return self._integration_element(codim)

    def volumes(self, codim):
        return self._volumes(codim)

    def volumes_inverse(self, codim):
        return self._volumes_inverse(codim)

    def unit_outer_normals(self):
        return self._unit_outer_normals()

    def centers(self, codim):
        return self._centers(codim)

    def diameters(self, codim):
        return self._diameters(codim)

    def quadrature_points(self, codim, order=None, npoints=None, quadrature_type='default'):
        return self._quadrature_points(codim, order, npoints, quadrature_type)
