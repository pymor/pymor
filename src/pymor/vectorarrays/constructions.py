# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import abstractmethod

from pymor.vectorarrays.interface import VectorSpace
from pymor.vectorarrays.block import BlockVectorArray
from pymor.vectorarrays.interface import VectorArray


def cat_arrays(vector_arrays):
    """Return a new |VectorArray| which is a concatenation of the arrays in `vector_arrays`."""
    vector_arrays = list(vector_arrays)
    total_length = sum(map(len, vector_arrays))
    cated_arrays = vector_arrays[0].empty(reserve=total_length)
    for a in vector_arrays:
        cated_arrays.append(a)
    return cated_arrays


class ProjectedVectorArray(VectorArray):
    """|VectorArray| class for vectors from the range of a projector.

    The `is_projected` attribute indicates wether the projector has already been
    applied.
    """

    def __init__(self, va, space, is_projected):
        assert isinstance(space, ProjectorSpace)
        self._va = va
        self.space = space
        self.is_projected = is_projected

    def get_va(self):
        """Get the |VectorArray| as an object from the super space."""
        if self.is_projected:
            return self._va
        else:
            self._va = self.space.apply_projector(self._va)._va
            self.is_projected = True
            return self._va

    def ones(self, count=1, reserve=0):
        # there is no guarantee that the one vector is in self.space
        raise NotImplementedError

    def __len__(self):
        return len(self._va)

    def __getitem__(self, ind):
        U = type(self)(self._va.__getitem__(ind), self.space, is_projected=self.is_projected)
        U.is_view = True
        return U

    def __delitem__(self, ind):
        self._va.__delitem__(ind)

    def to_numpy(self, ensure_copy=False):
        return self.get_va().to_numpy()

    def append(self, other, remove_from_other=False):
        if other in self.space:
            if self.is_projected == other.is_projected:
                self._va.append(other._va, remove_from_other=remove_from_other)
            elif self.is_projected:
                self._va.append(other.get_va(), remove_from_other=remove_from_other)
            else:
                self._va = self.space.project_onto_subspace(self._va)._va
                self.is_projected = True
                self._va.append(other._va, remove_from_other=remove_from_other)
        else:
            raise NotImplementedError

    def copy(self, deep=False):
        return type(self)(self._va.copy(deep=deep), self.space, is_projected=self.is_projected)

    def scal(self, alpha):
        self._va.scal(alpha)

    def axpy(self, alpha, x):
        if x in self.space:
            if self.is_projected == x.is_projected:
                self._va.axpy(alpha, x._va)
            elif self.is_projected:
                self._va.axpy(alpha, x.get_va())
            else:
                self._va = self.space.project_onto_subspace(self._va)._va
                self.is_projected = True
                self._va.axpy(alpha, x._va)
        else:
            raise NotImplementedError

    def inner(self, other, product=None):
        if other in self.space:
            return self.get_va().inner(other.get_va(), product=product)
        else:
            raise NotImplementedError

    def pairwise_inner(self, other, product=None):
        if other in self.space:
            return self.get_va().pairwise_inner(other.get_va(), product=product)
        else:
            raise NotImplementedError

    def lincomb(self, coefficients):
        return type(self)(self._va.lincomb(coefficients), self.space, is_projected=self.is_projected)

    def _norm(self):
        return self.get_va()._norm()

    def _norm2(self):
        return self.get_va()._norm2()

    def dofs(self, dof_indices):
        raise NotImplementedError

    def amax(self):
        return self.get_va().amax()

    @property
    def real(self):
        return type(self)(self._va.real, self.space, is_projected=self.is_projected)

    @property
    def imag(self):
        return type(self)(self._va.imag, self.space, is_projected=self.is_projected)

    def conj(self):
        return type(self)(self._va.conj(), self.space, is_projected=self.is_projected)


class ProjectorSpace(VectorSpace):
    """|VectorSpace| containing vectors from the range of a projector."""

    def __init__(self, super_space, dim, id=None):
        self.super_space = super_space
        self.dim = dim
        self.id = id

    def __eq__(self, other):
        return type(other) is type(self) and self.super_space == other.super_space \
            and self.dim == other.dim

    def zeros(self, count=1, reserve=0):
        return self.make_array(self.super_space.zeros(count, reserve), True)

    def random(self, count=1, distribution='uniform', random_state=None, seed=None, reserve=0, **kwargs):
        super_random = self.super_space.random(count, distribution, random_state, seed, reserve, **kwargs)
        return self.make_array(super_random, is_projected=False)

    @abstractmethod
    def make_array(self, super_va, is_projected):
        pass

    def from_numpy(self, V, ensure_copy=False):
        return self.make_array(self.super_space.from_numpy(V), is_projected=False)

    @abstractmethod
    def apply_projector(self, V):
        pass


class LerayProjectedVectorArray(ProjectedVectorArray):

    def __init__(self, va, space, is_projected):
        """|VectorArray| class for vectors from |LerayProjectorRangeSpace|s."""
        assert isinstance(space, LerayProjectorSpace)
        super().__init__(va, space, is_projected)


class LerayProjectorSpace(ProjectorSpace):
    """|VectorSpace| containing vectors from the range of a discrete Leray Projector.

    Based on linear |Operator|s :math:`E`, :math:`G` and :math:`J` the projectors

    .. math::
        P = I - E^{-1} G (J E^{-1} G)^{-1} J

    and

    .. math::
        Q = I - G (J E^{-1} G)^{-1} J E^{-1}

    can be defined. This |VectorSpace| represents the subspace of E.source
    where for all vectors :math:`V` it holds

    .. math::
        P V = V

    if `self.trans == False` and `self.range_space == True`,

    .. math::
        P^T V = V

    if `self.trans == True` and `self.range_space == True`,

    .. math::
        Q V = V

    if `self.trans == False` and `self.range_space == False` or

    .. math::
        Q^T V = V

    if `self.trans == True` and `self.range_space == False`.
    """

    def __init__(self, E, G, range_space=True, trans=False, id=None):
        self.E = E
        self.G = G
        self.range_space = range_space
        self.trans = trans
        dim = E.source.dim - G.source.dim if not trans else E.source.dim - G.range.dim
        super().__init__(E.source, dim, id=id)

    def __eq__(self, other):
        return type(other) is type(self) and self.E == other.E and self.G == other.G \
            and self.trans == other.trans and self.range_space == other.range_space

    def make_array(self, super_va, is_projected):
        return LerayProjectedVectorArray(super_va, self, is_projected)

    def apply_projector(self, V):
        assert V in self.super_space
        from pymor.operators.block import BlockOperator
        if self.range_space:
            sps = BlockOperator([
                [self.E, self.G],
                [self.G.H, None]
            ])
        else:
            # this has to be changed as soon as G is different from J^T
            sps = BlockOperator([
                [self.E, self.G],
                [self.G.H, None]
            ])
        if not self.trans:
            rhs = BlockVectorArray([V, self.G.source.zeros(len(V))])
            Vp = self.E.apply(sps.apply_inverse(rhs).block(0))
        else:
            rhs = BlockVectorArray([self.E.apply_adjoint(V), self.G.source.zeros(len(V))])
            Vp = sps.apply_inverse_adjoint(rhs).block(0)
        return self.make_array(Vp, is_projected=True)

    def trans_projector(self):
        """Return the |LerayProjectorSpace| w.r.t. the transposed projector."""
        return LerayProjectorSpace(self.E, self.G, not self.trans)
