# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

try:
    import ngsolve
    HAVE_NGSOLVE = True
except ImportError:
    HAVE_NGSOLVE = False

if HAVE_NGSOLVE:
    from ngsolve import GridFunction
    import numpy as np

    from pymor.vectorarrays.interfaces import VectorSpace
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorArray


    class NGSolveVector(CopyOnWriteVector):
        """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl
            self.vec = impl.vec

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            new_impl = GridFunction(self.impl.space)
            new_impl.vec.data = self.vec
            self.impl = new_impl
            self.vec = new_impl.vec

        @classmethod
        def make_zeros(cls, subtype):
            impl = GridFunction(subtype)  # subtype is FESpace
            return cls(impl)

        @property
        def dim(self):
            return self.vec.size

        @property
        def subtype(self):
            return self.impl.space

        @property
        def data(self):
            return self.vec.FV().NumPy()

        def _scal(self, alpha):
            self.vec.data = float(alpha) * self.vec

        def _axpy(self, alpha, x):
            self.vec.data = self.vec + float(alpha) * x.vec

        def dot(self, other):
            return self.vec.InnerProduct(other.vec)

        def l1_norm(self):
            return np.linalg.norm(self.data, ord=1)

        def l2_norm(self):
            return self.vec.Norm()

        def l2_norm2(self):
            return self.vec.Norm() ** 2

        def components(self, component_indices):
            if len(component_indices) == 0:
                return np.array([], dtype=np.intc)
            assert 0 <= np.min(component_indices)
            assert np.max(component_indices) < self.dim
            vec = self.vec
            return np.array([vec[int(i)] for i in component_indices])

        def amax(self):
            A = np.abs(self.data)
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val


    def NGSolveVectorSpace(fespace):
        return VectorSpace(ListVectorArray, (NGSolveVector, fespace))
