from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

# For python3.2 and greater, we user functools.lru_cache for caching. If our python
# version is to old, we import the same decorator from the third-party functools32
# package
try:
    from functools import lru_cache
except ImportError:
    from functools32 import lru_cache

import pymor.core as core
from pymor.core.exceptions import CodimError



class IConformalTopologicalGrid(core.BasicInterface):
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
    def size(self, codim=0):
        '''size(codim) is the number of entities in the grid of codimension codim'''
        pass

    @core.interfaces.abstractmethod
    @lru_cache(maxsize=None)
    def subentities(self, codim=0, subentity_codim=None):
        '''retval[e,s] is the global index of the s-th codim-"subentity_codim"
        subentity of the codim-"codim" entity with global index e.

        If subentity_codim=None, it is set to codim+1.

        Only subentities(codim, None) has to be implemented, by default,
        subentities(codim, subentity_codim) is computed by calculating the
        transitive closure of subentities(codim, None)
        '''
        assert 0 <= codim < self.dim, CodimError('Invalid codimension')
        if subentity_codim > codim + 1:
            SE = self.subentities(codim, subentity_codim - 1)
            SESE = self.subentities(subentity_codim - 1, subentity_codim)

            # we assume that there is only one geometry type ...
            num_subsubentities = np.unique(SESE[SE[0]]).size

            SSE = np.empty((SE.shape[0], num_subsubentities), dtype=np.int32)
            SSE.fill(-1)

            for ei in xrange(SE.shape[0]):
                X = SESE[SE[ei]].ravel()
                SSE[ei] = X[np.sort(np.unique(X, return_index=True)[1])]

            return SSE
        else:
            raise NotImplementedError

    @lru_cache(maxsize=None)
    def superentities(self, codim, superentity_codim=None):
        '''retval[e,s] is the global index of the s-th codim-"superentity_codim"
        superentity of the codim-"codim" entity with global index e.

        If superentity_codim == None, it is set to codim-1.

        The default implementation is to compute the result from subentities()
        '''
        assert 0 < codim <= self.dim, CodimError('Invalid codimension')
        if superentity_codim is None:
            superentity_codim = codim - 1

        SE = self.subentities(superentity_codim, codim)
        num_superentities = np.bincount(SE.ravel()).max()
        SPE = np.empty((self.size(codim), num_superentities), dtype=np.int32)
        SPE.fill(-1)

        SPE_COUNTS = np.zeros(SPE.shape[0], dtype=np.int32)

        for index, se in np.ndenumerate(SE):
            if se >= 0:
                SPE[se, SPE_COUNTS[se]] = index[0]
                SPE_COUNTS[se] += 1

        return SPE

    @lru_cache(maxsize=None)
    def neighbours(self, codim=0, neighbour_codim=0, intersection_codim=None):
        '''retval[e,s] is the global index of the n-th codim-"neighbour_codim"
        entitiy of the codim-"codim" entity with global index e that shares
        with it an intersection of codimension "intersection_codim".

        If intersection_codim == None,
            it is set to codim if codim == neighbour_codim
            otherwise it is set to min(codim, neighbour_codim).

        The default implementation is to compute the result from subentities()
        and superentities().
        '''
        if intersection_codim is None:
            if codim == neighbour_codim:
                intersection_codim = codim + 1
            else:
                intersection_codim = min(codim, neighbour_codim)

        if intersection_codim == min(codim, neighbour_codim):
            if codim <= neighbour_codim:
                return self.subentities(codim, neighbour_codim)
            else:
                return self.superentities(codim, neighbour_codim)
        else:
            EI = self.subentities(codim, intersection_codim)
            ISE = self.superentities(intersection_codim, neighbour_codim)

            NB = np.empty((EI.shape[0], EI.shape[1] * ISE.shape[1]), dtype=np.int32)
            NB.fill(-1)
            NB_COUNTS = np.zeros(EI.shape[0], dtype=np.int32)

            if codim == neighbour_codim:
                for ii, i in np.ndenumerate(EI):
                    if i >= 0:
                        for ni, n in np.ndenumerate(ISE[i]):
                            if n != ii[0] and n not in NB[ii[0]]:
                                NB[ii[0], NB_COUNTS[ii[0]]] = n
                                NB_COUNTS[ii[0]] += 1
            else:
                for ii, i in np.ndenumerate(EI):
                    if i >= 0:
                        for ni, n in np.ndenumerate(ISE[i]):
                            if n not in NB[ii[0]]:
                                NB[ii[0], NB_COUNTS[ii[0]]] = n
                                NB_COUNTS[ii[0]] += 1

            NB = NB[:NB.shape[0], :NB_COUNTS.max()]
            return NB


class ISimpleReferenceElement(core.BasicInterface):
    '''Defines a reference element with the property that each of its subentities is
    of the same type. I.e. a three-dimensional reference element cannot have triangles
    and rectangles as faces at the same time
    '''

    dim = None
    volume = None

    @core.interfaces.abstractmethod
    def size(self, codim=1):
        'Number of subentites of codimension "codim"'

    @core.interfaces.abstractmethod
    def subentities(self, codim, subentity_codim):
        '''subentities(c,sc)[i,j] is - with respect to the indexing inside the
        reference element  - the index of the i-th "subentity_codim"-codim
        subentity of the j-th "codim"-codim subentity of the reference element
        '''
        pass

    @core.interfaces.abstractmethod
    @lru_cache(maxsize=None)
    def subentity_embedding(self, subentity_codim):
        '''returns a tuple (A, B) which defines the embedding of the "subentity_codim"-
        subentity with index "index" into the reference element.
        for subsubentites, the embedding is by default given via its embedding into its
        lowest index superentity
        '''
        if subentity_codim > 1:
            A = []
            B = []
            for i in xrange(self.size(subentity_codim)):
                P = np.where(self.subentities(1, subentity_codim) == i)
                parent_index, local_index = P[0][0], P[1][0]
                A0, B0 = self.subentity_embedding(1)
                A0 = A0[parent_index]
                B0 = B0[parent_index]
                A1, B1 = self.sub_reference_element(1).subentity_embedding(subentity_codim-1)
                A1 = A1[local_index]
                B1 = B1[local_index]
                A.append(np.dot(A0, A1))
                B.append(np.dot(A0, B1) + B0)
            return np.array(A), np.array(B)
        else:
            raise NotImplementedError

    @core.interfaces.abstractmethod
    @lru_cache(maxsize=None)
    def sub_reference_element(self, codim=1):
        if subentity_codim > 1:
            return self.sub_reference_element(1).sub_reference_element(codim - 1)
        else:
            raise NotImplementedError

    @core.interfaces.abstractmethod
    def unit_outer_normals():
        pass

    @core.interfaces.abstractmethod
    def center():
        pass

    @core.interfaces.abstractmethod
    def mapped_diameter(A):
        pass
