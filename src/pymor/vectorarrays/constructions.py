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


class DivergenceFreeVectorArray(VectorArray):

    def __init__(self, va, space, is_projected=True):
        """|VectorArray| class for divergence free vectors.

        Class for |VectorArray|s from a |DivergenceFreeVectorSpace|. The parameter
        `is_projected` indicates wether the projection `P V` or `P^T V` has already
        been performed.
        """
        assert isinstance(space, DivergenceFreeSubSpace)
        self._va = va
        self.space = space
        self.is_projected = is_projected

    def get_va(self):
        """Get the |VectorArray| in coordinate space representation."""
        if self.is_projected:
            return self._va
        else:
            self._va = self.space.project_onto_subspace(self._va)._va
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
        return self.get_va().dofs(dof_indices)

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


class DivergenceFreeSubSpace(VectorSpace):
    """|VectorSpace| containing discrete divergence free velocities.

    Based on linear |Operator|s :math:`E` and :math:`G` the projector

    .. math::
        P = I - G (G^T E^{-1} G)^{-1} G^T E^{-1}

    can be defined. If `trans==True` this |VectorSpace| represents
    the subspace of E.source where for all vectors :math:`V` it holds

    .. math::
        P^T V = V.

    If `trans==False` it holds

    .. math::
        P V = V.
    """

    def __init__(self, E, G, trans, id=None):
        self.E = E
        self.G = G
        self.trans = trans
        self.coordinate_space = E.source
        self.dim = E.source.dim
        self.id = id

    def trans_projector(self):
        return type(self)(self.E, self.G, not self.trans, self.id)

    def __eq__(self, other):
        # this condition ignores self.trans, but leads to desired behaviour
        return type(other) is type(self) and self.E == other.E \
            and self.G == other.G  # and self.trans == other.trans

    def zeros(self, count=1, reserve=0):
        return DivergenceFreeVectorArray(self.coordinate_space.zeros(count, reserve), self)

    def random(self, count=1, distribution='uniform', random_state=None, seed=None, reserve=0, **kwargs):
        coordinate_random = self.coordinate_space.random(count, distribution, random_state, seed, reserve, **kwargs)
        return DivergenceFreeVectorArray(coordinate_random, self, is_projected=False)

    def make_array(*args, **kwargs):
        raise NotImplementedError

    def from_numpy(self, V, ensure_copy=False):
        from pymor.vectorarrays.numpy import NumpyVectorArray
        return DivergenceFreeVectorArray(NumpyVectorArray(V, self.coordinate_space), self)

    def project_onto_subspace(self, V):
        assert V in self.coordinate_space
        from pymor.operators.block import BlockOperator
        sps = BlockOperator([
            [self.E, self.G],
            [self.G.H, None]
        ])
        if self.trans:
            rhs = BlockVectorArray([self.E.apply_adjoint(V), self.G.source.zeros(len(V))])
            Vp = sps.apply_inverse_adjoint(rhs).block(0)
        else:
            rhs = BlockVectorArray([V, self.G.source.zeros(len(V))])
            Vp = self.E.apply(sps.apply_inverse(rhs).block(0))
        return DivergenceFreeVectorArray(Vp, self)
