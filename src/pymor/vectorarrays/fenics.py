# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    import dolfin as df
    HAVE_FENICS = True
    FENICS_VERSION = list(map(int, df.__version__.split('.')))
    if FENICS_VERSION[:2] != [1, 6]:
        import warnings
        warnings.warn('FEniCS support has only been tested with dolfin 1.6.')
except ImportError:
    HAVE_FENICS = False

if HAVE_FENICS:
    import numpy as np

    from pymor.vectorarrays.interfaces import VectorSpace
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorArray


    # For pyMOR's MPI support to work, subtypes need to be hashable.
    # However, FunctionSpace (our subtype) defines __eq__ but not __hash__
    # which makes it unhashable on Pyhton 3. On the other hand, equality
    # for FunctionSpace seems to be the same as identity, so we can easily
    # monkey-patch:
    df.FunctionSpace.__hash__ = lambda self: id(self)


    class FenicsVector(CopyOnWriteVector):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl, space):
            self.impl = impl
            self.space = space

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl, instance.space)

        def _copy_data(self):
            self.impl = self.impl.copy()

        @classmethod
        def make_zeros(cls, subtype):
            impl = df.Function(subtype).vector()
            return cls(impl, subtype)

        @property
        def dim(self):
            return self.impl.size()

        @property
        def subtype(self):
            return self.space

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

        def l2_norm2(self):
            return self.impl.norm('l2') ** 2

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


    def FenicsVectorSpace(V):
        return VectorSpace(ListVectorArray, (FenicsVector, V))
