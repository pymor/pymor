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
    from ngsolve import BaseVector
    import numpy as np

    from pymor.vectorarrays.interfaces import VectorSpace
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorArray


    class NGSolveVector(CopyOnWriteVector):
        """Wraps a NGSolve BaseVector to make it usable with ListVectorArray."""

        def __init__(self, impl):
            self.impl = impl
            self._array = impl.FV().NumPy()

        @classmethod
        def from_instance(cls, instance):
            return cls(instance.impl)

        def _copy_data(self):
            new_impl = BaseVector(self.impl.size)
            new_impl.data = self.impl
            self.impl = new_impl
            self._array = new_impl.FV().NumPy()

        @classmethod
        def make_zeros(cls, subtype):
            impl = BaseVector(subtype)
            impl.FV().NumPy()[:] = 0
            return cls(impl)

        @property
        def dim(self):
            return self.impl.size

        @property
        def subtype(self):
            return self.impl.size

        @property
        def data(self):
            return self._array

        def _scal(self, alpha):
            self.impl.data = float(alpha) * self.impl

        def _axpy(self, alpha, x):
            self.impl.data = self.impl + float(alpha) * x.impl

        def dot(self, other):
            return self.impl.InnerProduct(other.impl)

        def l1_norm(self):
            return np.linalg.norm(self._array, ord=1)

        def l2_norm(self):
            return self.impl.Norm()

        def l2_norm2(self):
            return self.impl.Norm() ** 2

        def components(self, component_indices):
            return self._array[component_indices]

        def amax(self):
            A = np.abs(self._array)
            max_ind = np.argmax(A)
            max_val = A[max_ind]
            return max_ind, max_val


    def NGSolveVectorSpace(dim):
        return VectorSpace(ListVectorArray, (NGSolveVector, dim))
