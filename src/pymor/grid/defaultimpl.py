from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pymor.core as core
from pymor.core.exceptions import CodimError


class IConformalTopologicalGridDefaultImplementation():

    @core.cached
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

    @core.cached
    def _superentities(self, codim, superentity_codim=None):
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

    @core.cached
    def _superentity_indices(self, codim, superentity_codim=None):
        assert 0 < codim <= self.dim, CodimError('Invalid codimension')
        if superentity_codim is None:
            superentity_codim = codim - 1
        E = self.subentities(superentity_codim, codim)
        SE = self.superentities(codim, superentity_codim)
        SEI = np.empty_like(SE)
        SEI.fill(-1)

        for index, e in np.ndenumerate(SE):
            if e >= 0:
                SEI[index] = np.where(E[e] == index[0])[0]

        return SEI

    @core.cached
    def _neighbours(self, codim, neighbour_codim, intersection_codim):
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

    @core.cached
    def _boundaries(self, codim):
        if codim == 1:
            SE = self.superentities(1, 0)
            # a codim-1 entity can have at most 2 superentities, and it is a boundary
            # if it has only one superentity
            return np.where(np.any(SE == -1, axis = 1))[0].astype('int32')
        elif codim == 0:
            B0 = self.boundaries(1)
            if B0.size > 0:
                return np.unique(self.superentities(1, 0)[B0])[1:]
            else:
                return np.array([])
        else:
            B0 = self.boundaries(1)
            if B0.size > 0:
                BC = np.unique(self.subentities(1, codim)[B0])
                if BC[0] == -1:
                    return BC[1:]
                else:
                    return BC
            else:
                return np.array([])

    @core.cached
    def _boundary_mask(self, codim):
        M = np.zeros(self.size(codim), dtype='bool')
        M[self.boundaries(codim)] = True
        return M


class ISimpleReferenceElementDefaultImplementation():

    @core.cached
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

    @core.cached
    def _sub_reference_element(self, codim):
        if codim > 1:
            return self.sub_reference_element(1).sub_reference_element(codim - 1)
        else:
            raise NotImplementedError


class ISimpleAffineGridDefaultImplementation():

    @core.cached
    def _subentities(self, codim, subentity_codim):
        assert 0 <= codim < self.dim, CodimError('Invalid codimension')
        assert 0 < codim, NotImplementedError
        P = self.superentities(codim, codim - 1)[:, 0]  # we assume here that superentites() is sorted by global index
        I = self.superentity_indices(codim, codim - 1)[:, 0]
        SE = self.subentities(codim - 1, subentity_codim)[P]
        RSE = self.reference_element(codim - 1).subentities(1, subentity_codim)[I]

        SSE = np.empty_like(RSE)
        for i in xrange(RSE.shape[0]):
            SSE[i, :] = SE[i, RSE[i]]

        return SSE

    @core.cached
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

    @core.cached
    def _jacobian_inverse_transposed(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        J = self.embeddings(codim)[0]
        JIT = np.array(map(np.linalg.pinv, J)).swapaxes(1, 2)
        return JIT

    @core.cached
    def _integration_element(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        J = self.embeddings(codim)[0]

        def f(A):
            return np.linalg.det(np.dot(A.T, A))

        V = np.array(map(f, J))
        return np.sqrt(V)

    @core.cached
    def _volumes(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        if codim == self.dim:
            return np.ones(self.size(self.dim))
        return self.reference_element(codim).volume * self.integration_element(codim)

    @core.cached
    def _volumes_inverse(self, codim):
        return np.reciprocal(self.volumes(codim))

    @core.cached
    def _unit_outer_normals(self):
        JIT = self.jacobian_inverse_transposed(0)
        N = np.dot(JIT, self.reference_element(0).unit_outer_normals().T).swapaxes(1, 2)
        return N / np.apply_along_axis(np.linalg.norm, 2, N)[:, :, np.newaxis]

    @core.cached
    def _centers(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        A, B = self.embeddings(codim)
        C = self.reference_element(codim).center()
        return np.dot(A, C) + B

    @core.cached
    def _diameters(self, codim):
        assert 0 <= codim <= self.dim,\
            CodimError('Invalid Codimension (must be between 0 and {} but was {})'.format(self.dim, codim))
        return np.squeeze(self.reference_element(codim).mapped_diameter(self.embeddings(codim)[0]))

    @core.cached
    def _quadrature_points(self, codim, order, npoints, quadrature_type):
        P, _ = self.reference_element(codim).quadrature(order, npoints, quadrature_type)
        A, B = self.embeddings(codim)
        return np.einsum('eij,kj->eki', A, P) + B[:, np.newaxis, :]
