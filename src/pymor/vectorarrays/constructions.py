# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

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

    If the `subspace` attribute is `None` this |VectorArray| represents the
    |VectorArray| specified via the `va` parameter. In this case the `subspace`
    attribute will derive dynamically (e.g. when appending to an empty |VectorArray|).
    The `is_projected` attribute indicates wether the projector has already been
    applied.
    """

    def __init__(self, va, subspace, is_projected):
        assert subspace is None or isinstance(subspace, ProjectorSpace)
        self._va = va
        self.space = va.space
        self.subspace = subspace
        self.is_projected = is_projected

    def get_va(self):
        """Get the |VectorArray| as an object from the super space."""
        if self.is_projected:
            return self._va
        else:
            self._va = self.subspace.apply_projector(self._va)._va
            self.is_projected = True
            return self._va

    def ones(self, count=1, reserve=0):
        # there is no guarantee that the one vector is in self.subspace
        raise NotImplementedError

    def __len__(self):
        return len(self._va)

    def __getitem__(self, ind):
        U = type(self)(self._va.__getitem__(ind), self.subspace, is_projected=self.is_projected)
        U.is_view = True
        return U

    def __delitem__(self, ind):
        self._va.__delitem__(ind)

    def to_numpy(self, ensure_copy=False):
        return self.get_va().to_numpy()

    def append(self, other, remove_from_other=False):
        assert other in self.space
        if len(self) == 0 and self.subspace is None:
            self.subspace = other.subspace
        if self.is_projected == other.is_projected:
            self._va.append(other._va, remove_from_other=remove_from_other)
        elif self.is_projected:
            self._va.append(other.get_va(), remove_from_other=remove_from_other)
        else:
            self._va = self.subspace.apply_projector(self._va)._va
            self.is_projected = True
            self._va.append(other._va, remove_from_other=remove_from_other)

    def copy(self, deep=False):
        return type(self)(self._va.copy(deep=deep), self.subspace, is_projected=self.is_projected)

    def scal(self, alpha):
        self._va.scal(alpha)

    def axpy(self, alpha, x):
        assert x in self.space and x.subspace == self.subspace
        if self.is_projected == x.is_projected:
            self._va.axpy(alpha, x._va)
        elif self.is_projected:
            self._va.axpy(alpha, x.get_va())
        else:
            self._va = self.subspace.apply_projector(self._va)._va
            self.is_projected = True
            self._va.axpy(alpha, x._va)

    def inner(self, other, product=None):
        assert other in self.space and other.subspace == self.subspace
        return self.get_va().inner(other.get_va(), product=product)

    def pairwise_inner(self, other, product=None):
        assert other in self.space and other.subspace == self.subspace
        return self.get_va().pairwise_inner(other.get_va(), product=product)

    def lincomb(self, coefficients):
        return type(self)(self._va.lincomb(coefficients), self.subspace, is_projected=self.is_projected)

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
        return type(self)(self._va.real, self.subspace, is_projected=self.is_projected)

    @property
    def imag(self):
        return type(self)(self._va.imag, self.subspace, is_projected=self.is_projected)

    def conj(self):
        return type(self)(self._va.conj(), self.subspace, is_projected=self.is_projected)


class ProjectorSpace(VectorSpace):
    """|VectorSpace| containing vectors from the range of a projector.

    This |VectorSpace| should only be used when it is necessary to create
    |ProjectedVectorArray|s whose subspace needs to be derived dynamically.
    """

    def __init__(self, super_space, id=None):
        self.super_space = super_space
        self.dim = super_space.dim
        self.id = id

    def __eq__(self, other):
        return type(other) is type(self) and self.super_space == other.super_space \
            and self.dim == other.dim

    def zeros(self, count=1, reserve=0):
        return self.make_array(self.super_space.zeros(count, reserve), True)

    def random(self, count=1, distribution='uniform', random_state=None, seed=None, reserve=0, **kwargs):
        super_random = self.super_space.random(count, distribution, random_state, seed, reserve, **kwargs)
        return self.make_array(super_random, is_projected=False)

    def make_array(self, super_va, is_projected):
        return ProjectedVectorArray(super_va, None, is_projected)

    def from_numpy(self, V, ensure_copy=False):
        return self.make_array(self.super_space.from_numpy(V), is_projected=False)

    def apply_projector(self, V):
        raise NotImplementedError

    def __contains__(self, other):
        return isinstance(other, ProjectedVectorArray) and self.super_space == getattr(other, 'space', None)


class LerayProjectorSpace(ProjectorSpace):
    """|VectorSpace| containing vectors from the range of a discrete Leray Projector.

    Based on linear |Operator|s :math:`E` and :math:`G` the projector

    .. math::
        P = I - E^{-1} G (G^T E^{-1} G)^{-1} G^T

    can be defined. This |VectorSpace| represents the subspace of E.source
    where for all vectors :math:`V` it holds

    .. math::
        P V = V

    if `self.trans == False` and

    .. math::
        P^T V = V

    if `self.trans == True`.
    """

    def __init__(self, E, G, trans, id=None):
        self.E = E
        self.G = G
        self.trans = trans
        super().__init__(E.source, id=id)

    def __eq__(self, other):
        return type(other) is type(self) and self.E == other.E and self.G == other.G \
            and self.trans == other.trans

    def apply_projector(self, V):
        assert V in self.super_space
        from pymor.operators.block import BlockOperator
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

    def make_array(self, super_va, is_projected):
        return ProjectedVectorArray(super_va, self, is_projected)
