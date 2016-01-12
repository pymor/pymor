# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

try:
    import dolfin as df
    HAVE_FENICS = True
except ImportError:
    HAVE_FENICS = False

if HAVE_FENICS:
    import numpy as np

    from pymor.core.defaults import defaults
    from pymor.vectorarrays.interfaces import VectorSpace
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorArray


    class FenicsVectorSubtype(tuple):

        def __eq__(self, other):
            return (type(other) is FenicsVectorSubtype and
                    df.cpp.common.MPI.size(self[0]) == df.cpp.common.MPI.size(other[0]) and
                    self[1] == other[1])


    class FenicsVector(CopyOnWriteVector):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            self.impl = self.impl.copy()

        def make_zeros(cls, subtype):
            impl = df.Vector(*subtype)
            return cls(impl)

        @property
        def dim(self):
            return self.impl.size()

        @property
        def subtype(self):
            impl = self.impl
            return FenicsVectorSubtype((impl.mpi_comm(), self.impl.size()))

        @property
        def data(self):
            return self.impl.array()  # WARNING: This creates a copy!

        def _scal(self, alpha):
            self.impl *= alpha

        def _axpy(self, alpha, x):
            if x is self:
                self.scal(1. + alpha)
            else:
                self.impl.axpy(alpha, x.impl)

        def dot(self, other):
            return self.impl.inner(other.impl)

        def l1_norm(self):
            return self.impl.norm('l1')

        def l2_norm(self):
            return self.impl.norm('l2')

        def sup_norm(self):
            return self.impl.norm('linf')

        def components(self, component_indices):
            component_indices = np.array(component_indices, dtype=np.intc)
            if len(component_indices) == 0:
                return np.array([], dtype=np.intc)
            assert 0 <= np.min(component_indices)
            assert np.max(component_indices) < self.dim
            x = df.Vector()
            self.impl.gather(x, component_indices)
            return x.array()

        def amax(self):
            A = np.abs(self.impl.array())  # there seems to be no way in the interface to
                                           # compute amax without making a copy. also,
                                           # we need to check how things behave in the MPI
                                           # parallel case.
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val

        def __add__(self, other):
            return FenicsVector(self.impl + other.impl)

        def __iadd__(self, other):
            self._copy_data_if_needed()
            self.impl += other.impl
            return self

        __radd__ = __add__

        def __sub__(self, other):
            return FenicsVector(self.impl - other.impl)

        def __isub__(self, other):
            self._copy_data_if_needed()
            self.impl -= other.impl
            return self

        def __mul__(self, other):
            return FenicsVector(self.impl * other)

        def __neg__(self):
            return FenicsVector(-self.impl)


    def FenicsVectorSpace(dim, mpi_comm=df.mpi_comm_world()):
        return VectorSpace(ListVectorArray, (FenicsVector, FenicsVectorSubtype((mpi_comm, dim))))
