# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

from pymor.tools import mpi
from pymor.vectorarrays.interfaces import VectorArrayInterface


class MPIVectorArray(VectorArrayInterface):

    def __init__(self, cls, subtype, obj_id):
        self.cls = cls
        self.array_subtype = subtype
        self.obj_id = obj_id

    @classmethod
    def make_array(cls, subtype, count=0, reserve=0):
        return MPIVectorArray(subtype[0], subtype[1],
                              mpi.call(_make_array, subtype[0], subtype=subtype[1], count=count, reserve=reserve))

    def __len__(self):
        return mpi.call(mpi.method_call, self.obj_id, '__len__')

    @property
    def dim(self):
        return mpi.call(_dim, self.obj_id)

    @property
    def subtype(self):
        return (self.cls, self.array_subtype)

    def copy(self, ind=None):
        return MPIVectorArray(self.cls, self.array_subtype,
                              mpi.call(mpi.method_call_manage, self.obj_id, 'copy', ind=ind))

    def append(self, other, o_ind=None, remove_from_other=False):
        mpi.call(mpi.method_call1, self.obj_id, 'append', other.obj_id,
                 o_ind=o_ind, remove_from_other=remove_from_other)

    def remove(self, ind=None):
        mpi.call(mpi.method_call, self.obj_id, 'remove', ind=ind)

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        mpi.call(mpi.method_call1, self.obj_id, 'replace', other.obj_id,
                 ind=ind, o_ind=o_ind, remove_from_other=remove_from_other)

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'almost_equal', other.obj_id,
                        ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)

    def scal(self, alpha, ind=None):
        mpi.call(mpi.method_call, self.obj_id, 'scal', alpha, ind=ind)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        mpi.call(_axpy, self.obj_id, alpha, x.obj_id, ind=ind, x_ind=x_ind)

    def dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'dot', other.obj_id, ind=ind, o_ind=o_ind)

    def pairwise_dot(self, other, ind=None, o_ind=None):
        return mpi.call(mpi.method_call1, self.obj_id, 'pairwise_dot', other.obj_id, ind=ind, o_ind=o_ind)

    def lincomb(self, coefficients, ind=None):
        return MPIVectorArray(self.cls, self.array_subtype,
                              mpi.call(mpi.method_call_manage, self.obj_id, 'lincomb', coefficients, ind=ind))

    def l1_norm(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l1_norm', ind=ind)

    def l2_norm(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'l2_norm', ind=ind)

    def components(self, component_indices, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'components', component_indices, ind=ind)

    def amax(self, ind=None):
        return mpi.call(mpi.method_call, self.obj_id, 'amax', ind=ind)

    def __del__(self):
        mpi.call(mpi.remove_object, self.obj_id)


def _make_array(cls, subtype=None, count=0, reserve=0):
    obj = cls.make_array(subtype=subtype, count=count, reserve=reserve)
    return mpi.manage_object(obj)


def _dim(obj_id):
    obj = mpi.get_object(obj_id)
    return obj.dim


def _axpy(obj_id, alpha, x_obj_id, ind=None, x_ind=None):
    obj = mpi.get_object(obj_id)
    x = mpi.get_object(x_obj_id)
    obj.axpy(alpha, x, ind=ind, x_ind=x_ind)
