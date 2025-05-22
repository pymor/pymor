# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.base import ImmutableObject, abstractmethod
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


def cat_arrays(vector_arrays):
    """Return a new |VectorArray| which is a concatenation of the arrays in `vector_arrays`."""
    vector_arrays = list(vector_arrays)
    total_length = sum(map(len, vector_arrays))
    cated_arrays = vector_arrays[0].empty(reserve=total_length)
    for a in vector_arrays:
        cated_arrays.append(a)
    return cated_arrays


class AbstractBasis(ImmutableObject):
    space = None
    reduced_space = None

    @abstractmethod
    def reconstruct(self, U):
        pass

    @abstractmethod
    def project(self, U):
        pass


class VectorArrayBasis(AbstractBasis):

    def __init__(self, basis, space=None, product=None):
        assert space is None or basis is None or basis in space
        space = basis.space if basis is not None else space
        assert product is None or product.source == product.range == space

        if basis is not None:
            basis = basis.copy()
        self.__auto_init(locals())
        self.reduced_space = NumpyVectorSpace(len(basis)) if basis is not None else space

    def reconstruct(self, U):
        assert U in self.reduced_space
        return U.copy() if self.basis is None else self.basis.lincomb(U.to_numpy())

    def project(self, U):
        assert U in self.space
        if self.basis is None:
            return U.copy()
        else:
            return self.reduced_space.make_array(self.basis.inner(U, product=self.product))


class BlockedBasis(AbstractBasis):

    def __init__(self, bases, spaces=None):
        assert all(b is not None for b in bases) or spaces is not None
        assert spaces is None or all(b is None or b in s for b, s in zip(bases, spaces))
        bases = tuple(b.copy() for b in bases)
        self.__auto_init(locals())
        self.space = BlockVectorSpace(spaces if spaces is not None else [b.space for b in bases])
        self.reduced_space = BlockVectorSpace(
            [NumpyVectorSpace(len(b)) if b is not None else s for b, s in zip(bases, spaces)]
        )

    def reconstruct(self, U):
        assert U in self.reduced_space
        UU = self.space.make_array(
            [b.lincomb(u.to_numpy()) if b is not None else u for b, u in zip(self.bases, U.blocks)]
        )
        return UU

    def project(self, U):
        assert U in self.space
        UU = self.reduced_space.make_arra(
            [s.make_array(b.inner(u)) if b is not None else u
             for s, b, u in zip(self.reduced_space.subspace, self.bases, U.blocks)]
        )
        return UU
