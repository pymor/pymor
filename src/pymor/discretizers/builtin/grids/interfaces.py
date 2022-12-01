# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.analyticalproblems.domaindescriptions import KNOWN_BOUNDARY_TYPES
from pymor.core.base import abstractmethod
from pymor.core.cache import CacheableObject, cached
from pymor.core.logger import getLogger
from pymor.discretizers.builtin.inverse import inv_transposed_two_by_two
from pymor.discretizers.builtin.relations import inverse_relation


class ReferenceElement(CacheableObject):
    """Defines a reference element.

    All reference elements have the property that all subentities of a given codimension are of the
    same type. I.e. a three-dimensional reference element cannot have triangles and rectangles as
    faces at the same time.

    Attributes
    ----------
    dim
        The dimension of the reference element
    volume
        The volume of the reference element
    """

    dim = None
    volume = None
    cache_region = 'memory'

    @abstractmethod
    def size(self, codim):
        """Number of subentities of codimension `codim`."""

    @abstractmethod
    def subentities(self, codim, subentity_codim):
        """Return subentities.

        `subentities(c,sc)[i,j]` is, with respect to the indexing inside the
        reference element, the index of the `j`-th codim-`subentity_codim`
        subentity of the `i`-th codim-`codim` subentity of the reference element.
        """
        pass

    @abstractmethod
    def subentity_embedding(self, subentity_codim):
        """Return subentity embedding.

        Returns a tuple `(A, B)` which defines the embedding of the codim-`subentity_codim`
        subentities into the reference element.

        For `subentity_codim > 1', the embedding is by default given recursively via
        `subentity_embedding(subentity_codim - 1)` and
        `sub_reference_element(subentity_codim - 1).subentity_embedding(1)` choosing always
        the superentity with smallest index.
        """
        return self._subentity_embedding(subentity_codim)

    @cached
    def _subentity_embedding(self, subentity_codim):
        if subentity_codim > 1:
            A = []
            B = []
            for i in range(self.size(subentity_codim)):
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

    @abstractmethod
    def sub_reference_element(self, codim):
        """Returns the reference element of the codim-`codim` subentities."""
        return self._sub_reference_element(codim)

    @cached
    def _sub_reference_element(self, codim):
        if codim > 1:
            return self.sub_reference_element(1).sub_reference_element(codim - 1)
        else:
            raise NotImplementedError

    def __call__(self, codim):
        """Returns the reference element of the codim-`codim` subentities."""
        return self.sub_reference_element(codim)

    @abstractmethod
    def unit_outer_normals(self):
        """`retval[e]` is the unit outer-normal vector to the codim-1 subentity with index `e`."""
        pass

    @abstractmethod
    def center(self):
        """Coordinates of the barycenter."""
        pass

    @abstractmethod
    def mapped_diameter(self, A):
        """Return the diameter after transformation.

        The diameter of the reference element after transforming it with the matrix `A`
        (vectorized).
        """
        pass

    @abstractmethod
    def quadrature(self, order=None, npoints=None, quadrature_type='default'):
        """Return quadrature points and weights.

        Returns tuple `(P, W)` where `P` is an array of quadrature points with corresponding weights
        `W`.

        The quadrature is of order `order` or has `npoints` integration points.
        """
        pass

    @abstractmethod
    def quadrature_info(self):
        """Return quadrature information.

        Returns a tuple of dicts `(O, N)` where `O[quadrature_type]` is a list
        of orders which are implemented for `quadrature_type` and `N[quadrature_type]`
        is a list of the corresponding numbers of integration points.
        """
        pass

    def quadrature_types(self):
        o, _ = self.quadrature_info()
        return frozenset(o.keys())


class Grid(CacheableObject):
    """Affine grid.

    Topological grid with geometry where each codim-0 entity is affinely mapped to the same
    |ReferenceElement|.

    The grid is completely determined via the subentity relation given by :meth:`~Grid.subentities`
    and the embeddings given by :meth:`~Grid.embeddings`. In addition, only :meth:`~Grid.size` and
    :meth:`~Grid.reference_element` have to be implemented.
    """

    cache_region = 'memory'

    # if relative difference between domain points gets too large or too small
    # reference mapping etc numerics fail due to limited precision
    MAX_DOMAIN_WIDTH = 1e12
    MIN_DOMAIN_WIDTH = 1e-12
    MAX_DOMAIN_RATIO = 1e6

    @abstractmethod
    def size(self, codim):
        """The number of entities of codimension `codim`."""
        pass

    @abstractmethod
    def subentities(self, codim, subentity_codim):
        """Return subentities.

        `retval[e,s]` is the global index of the `s`-th codim-`subentity_codim` subentity of the
        codim-`codim` entity with global index `e`.

        The ordering of `subentities(0, subentity_codim)[e]` has to correspond, w.r.t. the embedding
        of `e`, to the local ordering inside the reference element.

        For `codim > 0`, we provide a default implementation by calculating the subentities of `e`
        as follows:

            1. Find the `codim-1` parent entity `e_0` of `e` with minimal global index
            2. Lookup the local indices of the subentities of `e` inside `e_0` using the reference
               element.
            3. Map these local indices to global indices using
               `subentities(codim - 1, subentity_codim)`.

        This procedures assures that `subentities(codim, subentity_codim)[e]` has the right ordering
        w.r.t. the embedding determined by `e_0`, which agrees with what is returned by
        `embeddings(codim)`
        """
        return self._subentities(codim, subentity_codim)

    @cached
    def _subentities(self, codim, subentity_codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        assert 0 < codim, 'Not implemented'
        P = self.superentities(codim, codim - 1)[:, 0]  # we assume here that superentities() is sorted by global index
        I = self.superentity_indices(codim, codim - 1)[:, 0]
        SE = self.subentities(codim - 1, subentity_codim)[P]
        RSE = self.reference_element(codim - 1).subentities(1, subentity_codim - (codim - 1))[I]

        SSE = np.empty_like(RSE)
        for i in range(RSE.shape[0]):
            SSE[i, :] = SE[i, RSE[i]]

        return SSE

    def superentities(self, codim, superentity_codim):
        """Return superentities.

        `retval[e,s]` is the global index of the `s`-th codim-`superentity_codim` superentity of the
        codim-`codim` entity with global index `e`.

        `retval[e]` is sorted by global index.

        The default implementation is to compute the result from
        `subentities(superentity_codim, codim)`.
        """
        return self._superentities(codim, superentity_codim)

    @cached
    def _superentities(self, codim, superentity_codim):
        return self._superentities_with_indices(codim, superentity_codim)[0]

    def superentity_indices(self, codim, superentity_codim):
        """Return indices of superentities.

        `retval[e,s]` is the local index of the codim-`codim` entity `e` in the
        codim-`superentity_codim` superentity `superentities(codim, superentity_codim)[e,s].`
        """
        return self._superentity_indices(codim, superentity_codim)

    @cached
    def _superentity_indices(self, codim, superentity_codim):
        return self._superentities_with_indices(codim, superentity_codim)[1]

    @cached
    def _superentities_with_indices(self, codim, superentity_codim):
        assert 0 <= codim <= self.dim, f'Invalid codimension (was {codim})'
        assert 0 <= superentity_codim <= codim, f'Invalid codimension (was {superentity_codim})'
        SE = self.subentities(superentity_codim, codim)
        return inverse_relation(SE, size_rhs=self.size(codim), with_indices=True)

    def neighbours(self, codim, neighbour_codim, intersection_codim=None):
        """Maps entity index and local neighbour index to global neighbour index

        `retval[e,n]` is the global index of the `n`-th codim-`neighbour_codim` entity of the
        codim-`codim` entity `e` that shares with `e` a subentity of codimension
        `intersection_codim`.

        If `intersection_codim == None`, it is set to `codim + 1` if `codim == neighbour_codim` and
        to `min(codim, neighbour_codim)` otherwise.

        The default implementation is to compute the result from
        `subentities(codim, intersection_codim)` and
        `superentities(intersection_codim, neihbour_codim)`.
        """
        return self._neighbours(codim, neighbour_codim, intersection_codim)

    @cached
    def _neighbours(self, codim, neighbour_codim, intersection_codim):
        assert 0 <= codim <= self.dim, 'Invalid codimension'
        assert 0 <= neighbour_codim <= self.dim, 'Invalid codimension'
        if intersection_codim is None:
            if codim == neighbour_codim:
                intersection_codim = codim + 1
            else:
                intersection_codim = max(codim, neighbour_codim)
        assert max(codim, neighbour_codim) <= intersection_codim <= self.dim, 'Invalid codimension'

        if intersection_codim == max(codim, neighbour_codim):
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

    def boundary_mask(self, codim):
        """Return boundary mask.

        `retval[e]` is true iff the codim-`codim` entity with global index `e` is a boundary entity.

        By definition, a codim-1 entity is a boundary entity if it has only one codim-0 superentity.
        For `codim != 1`, a codim-`codim` entity is a boundary entity if it has a codim-1
        sub/super-entity.
        """
        return self._boundary_mask(codim)

    @cached
    def _boundary_mask(self, codim):
        M = np.zeros(self.size(codim), dtype='bool')
        B = self.boundaries(codim)
        if B.size > 0:
            M[self.boundaries(codim)] = True
        return M

    def boundaries(self, codim):
        """Returns the global indices of all codim-`codim` boundary entities.

        By definition, a codim-1 entity is a boundary entity if it has only one codim-0 superentity.
        For `codim != 1`, a codim-`codim` entity is a boundary entity if it has a codim-1
        sub/super-entity.
        """
        return self._boundaries(codim)

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

    @abstractmethod
    def reference_element(self, codim):
        """The |ReferenceElement| of the codim-`codim` entities."""
        pass

    @abstractmethod
    def embeddings(self, codim):
        """Return embeddings.

        Returns tuple `(A, B)` where `A[e]` and `B[e]` are the linear part and the translation
        part of the map from the reference element of `e` to `e`.

        For `codim > 0`, we provide a default implementation by taking the embedding of the codim-1
        parent entity `e_0` of `e` with lowest global index and composing it with the
        subentity_embedding of `e` into `e_0` determined by the reference element.
        """
        return self._embeddings(codim)

    @cached
    def _embeddings(self, codim):
        assert codim > 0, NotImplemented
        E = self.superentities(codim, codim - 1)[:, 0]
        I = self.superentity_indices(codim, codim - 1)[:, 0]
        A0, B0 = self.embeddings(codim - 1)
        A0 = A0[E]
        B0 = B0[E]
        A1, B1 = self.reference_element(codim - 1).subentity_embedding(1)
        A = np.zeros((E.shape[0], A0.shape[1], A1.shape[2]))
        B = np.zeros((E.shape[0], A0.shape[1]))
        for i in range(A1.shape[0]):
            INDS = np.where(I == i)[0]
            A[INDS] = np.dot(A0[INDS], A1[i])
            B[INDS] = np.dot(A0[INDS], B1[i]) + B0[INDS]
        return A, B

    def jacobian_inverse_transposed(self, codim):
        """Return the inverse of transposed Jacobian.

        `retval[e]` is the transposed (pseudo-)inverse of the Jacobian of `embeddings(codim)[e]`.
        """
        return self._jacobian_inverse_transposed(codim)

    @cached
    def _jacobian_inverse_transposed(self, codim):
        assert 0 <= codim < self.dim,\
            f'Invalid Codimension (must be between 0 and {self.dim} but was {codim})'
        J = self.embeddings(codim)[0]
        if J.shape[-1] == J.shape[-2] == 2:
            JIT = inv_transposed_two_by_two(J)
        else:
            pinv = np.linalg.pinv
            JIT = np.array([pinv(j) for j in J]).swapaxes(1, 2)
        return JIT

    def integration_elements(self, codim):
        """`retval[e]` is given as `sqrt(det(A^T*A))`, where `A = embeddings(codim)[0][e]`."""
        return self._integration_elements(codim)

    @cached
    def _integration_elements(self, codim):
        assert 0 <= codim <= self.dim,\
            f'Invalid Codimension (must be between 0 and {self.dim} but was {codim})'

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
            D = np.array([f(j) for j in J])

        return np.sqrt(D)

    def volumes(self, codim):
        """Return volumes.

        `retval[e]` is the (dim-`codim`)-dimensional volume of the codim-`codim` entity with global
        index `e`.
        """
        return self._volumes(codim)

    @cached
    def _volumes(self, codim):
        assert 0 <= codim <= self.dim,\
            f'Invalid Codimension (must be between 0 and {self.dim} but was {codim})'
        if codim == self.dim:
            return np.ones(self.size(self.dim))
        return self.reference_element(codim).volume * self.integration_elements(codim)

    def volumes_inverse(self, codim):
        """`retval[e] = 1 / volumes(codim)[e]`."""
        return self._volumes_inverse(codim)

    @cached
    def _volumes_inverse(self, codim):
        return np.reciprocal(self.volumes(codim))

    def unit_outer_normals(self):
        """Return unit outer normals.

        `retval[e,i]` is the unit outer normal to the i-th codim-1 subentity of the codim-0 entity
        with global index `e`.
        """
        return self._unit_outer_normals()

    @cached
    def _unit_outer_normals(self):
        JIT = self.jacobian_inverse_transposed(0)
        N = np.dot(JIT, self.reference_element(0).unit_outer_normals().T).swapaxes(1, 2)
        return N / np.apply_along_axis(np.linalg.norm, 2, N)[:, :, np.newaxis]

    def centers(self, codim):
        """`retval[e]` is the barycenter of the codim-`codim` entity with global index `e`."""
        return self._centers(codim)

    @cached
    def _centers(self, codim):
        assert 0 <= codim <= self.dim,\
            f'Invalid Codimension (must be between 0 and {self.dim} but was {codim})'
        A, B = self.embeddings(codim)
        C = self.reference_element(codim).center()
        return np.dot(A, C) + B

    def diameters(self, codim):
        """`retval[e]` is the diameter of the codim-`codim` entity with global index `e`."""
        return self._diameters(codim)

    @cached
    def _diameters(self, codim):
        assert 0 <= codim <= self.dim,\
            f'Invalid Codimension (must be between 0 and {self.dim} but was {codim})'
        return np.reshape(self.reference_element(codim).mapped_diameter(self.embeddings(codim)[0]), (-1,))

    def quadrature_points(self, codim, order=None, npoints=None, quadrature_type='default'):
        """`retval[e]` is an array of quadrature points in global coordinates for the codim-`codim`
        entity with global index `e`.

        The quadrature is of order `order` or has `npoints` integration points. To integrate a
        function `f` over `e` one has to form ::

            np.dot(f(quadrature_points(codim, order)[e]),
                reference_element(codim).quadrature(order)[1]) *
            integration_elements(codim)[e].  # NOQA
        """
        return self._quadrature_points(codim, order, npoints, quadrature_type)

    @cached
    def _quadrature_points(self, codim, order, npoints, quadrature_type):
        P, _ = self.reference_element(codim).quadrature(order, npoints, quadrature_type)
        A, B = self.embeddings(codim)
        return np.einsum('eij,kj->eki', A, P) + B[:, np.newaxis, :]

    def bounding_box(self):
        """Returns a `(2, dim)`-shaped array containing lower/upper bounding box coordinates."""
        return self._bounding_box()

    @cached
    def _bounding_box(self):
        bbox = np.empty((2, self.dim))
        centers = self.centers(self.dim)
        for dim in range(self.dim):
            bbox[0, dim] = np.min(centers[:, dim])
            bbox[1, dim] = np.max(centers[:, dim])
        return bbox

    @classmethod
    def _check_domain(cls, domain):
        ll, rr = np.array(domain[0]), np.array(domain[1])
        sizes = rr - ll
        logger = getLogger('pymor.discretizers.builtin.grid')
        if np.linalg.norm(sizes) > cls.MAX_DOMAIN_WIDTH:
            logger.warning(f'Domain {domain} for {cls} exceeds width limit. Results may be inaccurate')
            return False
        if np.max(sizes) / np.min(sizes) > cls.MAX_DOMAIN_RATIO:
            logger.warning(f'Domain {domain} for {cls} exceeds ratio limit. Results may be inaccurate')
            return False
        return True


class GridWithOrthogonalCenters(Grid):
    """|Grid| with an additional `orthogonal_centers` method."""

    @abstractmethod
    def orthogonal_centers(self):
        """Return orthogonal centers.

        `retval[e]` is a point inside the codim-0 entity with global index `e` such that the line
        segment from `retval[e]` to `retval[e2]` is always orthogonal to the codim-1 entity shared
        by the codim-0 entities with global index `e` and `e2`.

        (This is mainly useful for gradient approximation in finite volume schemes.)
        """
        pass


class BoundaryInfo(CacheableObject):
    """Provides boundary types for the boundaries of a given |Grid|.

    For every boundary type and codimension a mask is provided, marking grid entities of the
    respective type and codimension by their global index.

    Attributes
    ----------
    boundary_types
        set of all boundary types the grid has.
    """

    boundary_types = frozenset()
    cache_region = 'memory'

    def mask(self, boundary_type, codim):
        """Return mask.

        retval[i] is `True` if the codim-`codim` entity of global index `i` is associated to the
        boundary type `boundary_type`.
        """
        raise ValueError(f'Has no boundary_type "{boundary_type}"')

    def unique_boundary_type_mask(self, codim):
        """Return unique boundary type mask.

        retval[i] is `True` if the codim-`codim` entity of global index `i` is associated to one
        and only one boundary type.
        """
        return np.less_equal(sum(self.mask(bt, codim=codim).astype(int) for bt in self.boundary_types), 1)

    def no_boundary_type_mask(self, codim):
        """Return no boundary type mask.

        retval[i] is `True` if the codim-`codim` entity of global index `i` is associated to no
        boundary type.
        """
        return np.equal(sum(self.mask(bt, codim=codim).astype(int) for bt in self.boundary_types), 0)

    def check_boundary_types(self, assert_unique_type=(1,), assert_some_type=()):
        for bt in self.boundary_types:
            if bt not in KNOWN_BOUNDARY_TYPES:
                self.logger.warning(f'Unknown boundary type: {bt}')

        if assert_unique_type:
            for codim in assert_unique_type:
                assert np.all(self.unique_boundary_type_mask(codim))
        if assert_some_type:
            for codim in assert_some_type:
                assert not np.any(self.no_boundary_type_mask(codim))

    @property
    def has_dirichlet(self):
        return 'dirichlet' in self.boundary_types

    @property
    def has_neumann(self):
        return 'neumann' in self.boundary_types

    @property
    def has_robin(self):
        return 'robin' in self.boundary_types

    def dirichlet_mask(self, codim):
        return self.mask('dirichlet', codim)

    def neumann_mask(self, codim):
        return self.mask('neumann', codim)

    def robin_mask(self, codim):
        return self.mask('robin', codim)

    @cached
    def _boundaries(self, boundary_type, codim):
        return np.where(self.mask(boundary_type, codim))[0].astype('int32')

    def boundaries(self, boundary_type, codim):
        return self._boundaries(boundary_type, codim)

    def dirichlet_boundaries(self, codim):
        return self._boundaries('dirichlet', codim)

    def neumann_boundaries(self, codim):
        return self._boundaries('neumann', codim)

    def robin_boundaries(self, codim):
        return self._boundaries('robin', codim)
