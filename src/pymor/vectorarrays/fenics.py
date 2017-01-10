# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_FENICS:
    import dolfin as df
    import numpy as np

    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace


    class FenicsVector(CopyOnWriteVector):
        """Wraps a FEniCS vector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            self.impl = self.impl.copy()

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
            assert np.max(component_indices) < self.impl.size()
            x = df.Vector()
            self.impl.gather(x, component_indices)
            return x.array()

        def amax(self):
            A = np.abs(self.impl.array())
            # there seems to be no way in the interface to compute amax without making a copy. also,
            # we need to check how things behave in the MPI parallel case.
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


    class FenicsVectorSpace(ListVectorSpace):

        def __init__(self, V, id_='STATE'):
            self.V = V
            self.id = id_

        @property
        def dim(self):
            return df.Function(self.V).vector().size()

        def __eq__(self, other):
            return type(other) is FenicsVectorSpace and self.V == other.V and self.id == other.id

        # since we implement __eq__, we also need to implement __hash__
        def __hash__(self):
            return id(self.V) + hash(self.id)

        def zero_vector(self):
            impl = df.Function(self.V).vector()
            return FenicsVector(impl)

        def make_vector(self, obj):
            return FenicsVector(obj)
