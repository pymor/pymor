# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core.cache import Cachable, cached
from pymor.core.exceptions import CodimError
from pymor.la.inverse import inv_transposed_two_by_two
from pymor.tools.relations import inverse_relation


class ConformalTopologicalGridDefaultImplementations(Cachable):

    @cached
    def _subentities(self, codim, subentity_codim=None):
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

    @cached
    def _superentities_with_indices(self, codim, superentity_codim=None):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension (was {})'.format(codim))
        if superentity_codim is None:
            superentity_codim = codim - 1 if codim > 0 else 0
        assert 0 <= superentity_codim <= codim, CodimError('Invalid codimension (was {})'.format(superentity_codim))
        SE = self.subentities(superentity_codim, codim)
        return inverse_relation(SE, size_rhs=self.size(codim), with_indices=True)

    @cached
    def _superentities(self, codim, superentity_codim=None):
        return self._superentities_with_indices(codim, superentity_codim)[0]

    @cached
    def _superentity_indices(self, codim, superentity_codim=None):
        return self._superentities_with_indices(codim, superentity_codim)[1]

    @cached
    def _neighbours(self, codim, neighbour_codim, intersection_codim):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension')
        assert 0 <= neighbour_codim <= self.dim, CodimError('Invalid codimension')
        if intersection_codim is None:
            if codim == neighbour_codim:
                intersection_codim = codim + 1
            else:
                intersection_codim = min(codim, neighbour_codim)
        assert max(codim, neighbour_codim) <= intersection_codim <= self.dim, CodimError('Invalid codimension')

        if intersection_codim == min(codim, neighbour_codim):
            if codim < neighbour_codim:
                return self.subentities(codim, neighbour_codim)
            elif codim > neighbour_codim:
                return self.superentities(codim, neighbour_codim)
            else:
                return np.zeros((self.size(codim), 0), dtype=np.int32)
        else:
            EI = self.subentities(codim, intersection_codim)
            ISE = self.superentities(intersection_codim, neighbour_codim)

            NB = np.empty((EI.shape[0], EI.shape[1] * ISE.shape[1]), dtype=np.int32)
            NB.fill(-1)
            NB_COUNTS = np.zeros(EI.shape[0], dtype=np.int32)

            if codim == neighbour_codim:
                for ii, i in np.ndenumerate(EI):
                    if i >= 0:
                        for _, n in np.ndenumerate(ISE[i]):
                            if n != ii[0] and n not in NB[ii[0]]:
                                NB[ii[0], NB_COUNTS[ii[0]]] = n
                                NB_COUNTS[ii[0]] += 1
            else:
                for ii, i in np.ndenumerate(EI):
                    if i >= 0:
                        for _, n in np.ndenumerate(ISE[i]):
                            if n not in NB[ii[0]]:
                                NB[ii[0], NB_COUNTS[ii[0]]] = n
                                NB_COUNTS[ii[0]] += 1

            NB = NB[:NB.shape[0], :NB_COUNTS.max()]
            return NB

    @cached
    def _boundaries(self, codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        if codim == 1:
            SE = self.superentities(1, 0)
            # a codim-1 entity can have at most 2 superentities, and it is a boundary
            # if it has only one superentity
            if SE.shape[1] > 1:
                return np.where(np.any(SE == -1, axis=1))[0].astype('int32')
            else:
                return np.arange(SE.shape[0], dtype='int32')
        elif codim == 0:
            B1 = self.boundaries(1)
            if B1.size > 0:
                B0 = np.unique(self.superentities(1, 0)[B1])
                return B0[1:] if B0[0] == -1 else B0
            else:
                return np.array([], dtype=np.int32)
        else:
            B1 = self.boundaries(1)
            if B1.size > 0:
                BC = np.unique(self.subentities(1, codim)[B1])
                return BC[1:] if BC[0] == -1 else BC
            else:
                return np.array([], dtype=np.int32)

    @cached
    def _boundary_mask(self, codim):
        M = np.zeros(self.size(codim), dtype='bool')
        B = self.boundaries(codim)
        if B.size > 0:
            M[self.boundaries(codim)] = True
        return M


class SimpleReferenceElementDefaultImplementations(Cachable):

    @cached
    def _subentity_embedding(self, subentity_codim):
        if subentity_codim > 1:
            A = []
            B = []
            for i in xrange(self.size(subentity_codim)):
                P = np.where(self.subentities(subentity_codim - 1, subentity_codim) == i)
                parent_index, local_index = P[0][0], P[1][0]
                A0, B0 = self.subentity_embedding(subentity_codim - 1)
                A0 = A0[parent_index]
                B0 = B0[parent_index]
                A1, B1 = self.sub_reference_element(subentity_codim - 1).subentity_embedding(1)
                A1 = A1[local_index]
                B1 = B1[local_index]
                A.append(np.dot(A0, A1))
                B.append(np.dot(A0, B1) + B0)
            return np.array(A), np.array(B)
        else:
            raise NotImplementedError

    @cached
    def _sub_reference_element(self, codim):
        if codim > 1:
            return self.sub_reference_element(1).sub_reference_element(codim - 1)
        else:
            raise NotImplementedError


class AffineGridDefaultImplementations(object):

    @cached
    def _subentities(self, codim, subentity_codim):
        assert 0 <= codim <= self.dim, CodimError('Invalid codimension')
        assert 0 < codim, NotImplementedError
        P = self.superentities(codim, codim - 1)[:, 0]  # we assume here that superentites() is sorted by global index
        I = self.superentity_indices(codim, codim - 1)[:, 0]
        SE = self.subentities(codim - 1, subentity_codim)[P]
        RSE = self.reference_element(codim - 1).subentities(1, subentity_codim - (codim - 1))[I]

        SSE = np.empty_like(RSE)
        for i in xrange(RSE.shape[0]):
            SSE[i, :] = SE[i, RSE[i]]

        return SSE

    @cached
    def _embeddings(self, codim=0):
        assert codim > 0, NotImplemented
        E = self.superentities(codim, codim - 1)[:, 0]
        I = self.superentity_indices(codim, codim - 1)[:, 0]
        A0, B0 = self.embeddings(codim - 1)
        A0 = A0[E]
        B0 = B0[E]
        A1, B1 = self.reference_element(codim - 1).subentity_embedding(1)
        A = np.zeros((E.shape[0], A0.shape[1], A1.shape[2]))
        B = np.zeros((E.shape[0], A0.shape[1]))
        for i in xrange(A1.shape[0]):
            INDS = np.where(I == i)[0]
            A[INDS] = np.dot(A0[INDS], A1[i])
            B[INDS] = np.dot(A0[INDS], B1[i]) + B0[INDS]
        return A, B

    @cached
    def _jacobian_inverse_transposed(self, codim):
        assert 0 <= codim < self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        J = self.embeddings(codim)[0]
        if J.shape[-1] == J.shape[-2] == 2:
            JIT = inv_transposed_two_by_two(J)
        else:
            JIT = np.array(map(np.linalg.pinv, J)).swapaxes(1, 2)
        return JIT

    @cached
    def _integration_elements(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))

        if codim == self.dim:
            return np.ones(self.size(codim))

        J = self.embeddings(codim)[0]
        JTJ = np.einsum('eji,ejk->eik', J, J)

        if JTJ.shape[1] == 1:
            D = JTJ.ravel()
        elif JTJ.shape[1] == 2:
            D = (JTJ[:, 0, 0] * JTJ[:, 1, 1] - JTJ[:, 1, 0] * JTJ[:, 0, 1]).ravel()
        else:
            def f(A):
                return np.linalg.det(A)
            D = np.array(map(f, J))

        return np.sqrt(D)

    @cached
    def _volumes(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        if codim == self.dim:
            return np.ones(self.size(self.dim))
        return self.reference_element(codim).volume * self.integration_elements(codim)

    @cached
    def _volumes_inverse(self, codim):
        return np.reciprocal(self.volumes(codim))

    @cached
    def _unit_outer_normals(self):
        JIT = self.jacobian_inverse_transposed(0)
        N = np.dot(JIT, self.reference_element(0).unit_outer_normals().T).swapaxes(1, 2)
        return N / np.apply_along_axis(np.linalg.norm, 2, N)[:, :, np.newaxis]

    @cached
    def _centers(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        A, B = self.embeddings(codim)
        C = self.reference_element(codim).center()
        return np.dot(A, C) + B

    @cached
    def _diameters(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        return np.reshape(self.reference_element(codim).mapped_diameter(self.embeddings(codim)[0]), (-1,))

    @cached
    def _quadrature_points(self, codim, order, npoints, quadrature_type):
        P, _ = self.reference_element(codim).quadrature(order, npoints, quadrature_type)
        A, B = self.embeddings(codim)
        return np.einsum('eij,kj->eki', A, P) + B[:, np.newaxis, :]
