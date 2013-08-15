# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)


from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.la import VectorArrayInterface
from pymor.operators import OperatorInterface

wrapped_vector_arrays = {}

def wrap_remote_vector_array(remote_view, remote_id):

    global wrapped_vector_arrays

    if (id(remote_view), remote_id) in wrapped_vector_arrays:
        return wrapped_vector_arrays[(id(remote_view), remote_id)]

    class WrappedRemoteVectorArray(RemoteVectorArray):
        rv = remote_view
        class_rid = remote_id

    WrappedRemoteVectorArray.__name__ = 'RemoteVectorArray_{}'.format(remote_id)
    wrapped_vector_arrays[(id(remote_view), remote_id)] = WrappedRemoteVectorArray
    return WrappedRemoteVectorArray


class RemoteVectorArray(VectorArrayInterface):

    class_rid = None

    @staticmethod
    def _empty(rid, dim, reserve=0):
        global RR
        U = RR[rid].empty(dim, reserve=reserve)
        U_id = id(U)
        RR[U_id] = U
        return U_id

    @classmethod
    def empty(cls, dim, reserve=0):
        U_id = cls.rv.apply(cls._empty, cls.class_rid, dim, reserve=reserve)
        return cls(U_id)

    @staticmethod
    def _zeros(rid, dim, count=1):
        global RR
        U = RR[rid].zeros(dim, count=count)
        U_id = id(U)
        RR[U_id] = U
        return U_id

    @classmethod
    def zeros(cls, dim, count=1):
        U_id = cls.rv.apply(cls._zeros, cls.class_rid, dim)
        return cls(U_id)

    def __init__(self, rid):
        self.rid = rid

    @staticmethod
    def _del(rid):
        global RR
        del RR[rid]

    def __del__(self):
        self.rv.apply(self._del, self.rid)

    @staticmethod
    def _len(rid):
        global RR
        return len(RR[rid])

    def __len__(self):
        return self.rv.apply(self._len, self.rid)

    @staticmethod
    def _dim(rid):
        global RR
        return RR[rid].dim

    @property
    def dim(self):
        return self.rv.apply(self._dim, self.rid)

    @staticmethod
    def _copy(rid, ind=None):
        global RR
        U = RR[rid].copy(ind)
        U_id = id(U)
        RR[U_id] = U
        return U_id

    def copy(self, ind=None):
        U_id = self.rv.apply(self._copy, self.rid, ind=ind)
        return type(self)(U_id)

    @staticmethod
    def _append(rid, other, o_ind=None, remove_from_other=False):
        global RR
        RR[rid].append(RR[other], o_ind=o_ind, remove_from_other=remove_from_other)

    def append(self, other, o_ind=None, remove_from_other=False):
        self.rv.apply(self._append, self.rid, other.rid, o_ind=o_ind, remove_from_other=remove_from_other)

    @staticmethod
    def _remove(rid, ind):
        global RR
        RR[rid].remove(ind)

    def remove(self, ind):
        self.rv.apply(self._remove, self.rid, ind=ind)

    @staticmethod
    def _replace(rid, other, ind=None, o_ind=None, remove_from_other=False):
        global RR
        RR[rid].replace(RR[other], ind=ind, o_ind=o_ind, remove_from_other=remove_from_other)

    def replace(self, other, ind=None, o_ind=None, remove_from_other=False):
        self.rv.apply(self._replace, self.rid, other.rid, ind=ind, o_ind=o_ind, remove_from_other=remove_from_other)

    @staticmethod
    def _almost_equal(rid, other, ind=None, o_ind=None, rtol=None, atol=None):
        global RR
        return RR[rid].almost_equal(other, ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)

    def almost_equal(self, other, ind=None, o_ind=None, rtol=None, atol=None):
        return self.rv.apply(self._almost_equal, self.rid, other.rid, ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)

    @staticmethod
    def _scal(rid, alpha, ind=None):
        global RR
        RR[rid].scal(alpha, ind=ind)

    def scal(self, alpha, ind=None):
        self.rv.apply(self._scal, self.rid, alpha, ind=ind)

    @staticmethod
    def _axpy(rid, alpha, x, ind=None, x_ind=None):
        global RR
        RR[rid].axpy(alpha, RR[x], ind=ind, x_ind=x_ind)

    def axpy(self, alpha, x, ind=None, x_ind=None):
        self.rv.apply(self._axpy, self.rid, alpha, x.rid, ind=ind, x_ind=x_ind)

    @staticmethod
    def _dot(rid, other, pairwise, ind=None, o_ind=None):
        global RR
        return RR[rid].dot(RR[other], pairwise, ind=ind, o_ind=o_ind)

    def dot(self, other, pairwise, ind=None, o_ind=None):
        return self.rv.apply(self._dot, self.rid, other.rid, pairwise, ind=ind, o_ind=o_ind)

    @staticmethod
    def _lincomb(rid, coefficients, ind=None):
        global RR
        U = RR[rid].lincomb(coefficients, ind=ind)
        U_id = id(U)
        RR[U_id] = U
        return U_id

    def lincomb(self, coefficients, ind=None):
        U_id = self.rv.apply(self._lincomb, self.rid, coefficients, ind=ind)
        return type(self)(U_id)

    @staticmethod
    def _l1_norm(rid, ind=None):
        global RR
        return RR[rid].l1_norm(ind=ind)

    def l1_norm(self, ind=None):
        return self.rv.apply(self._l1_norm, self.rid, ind=ind)

    @staticmethod
    def _l2_norm(rid, ind=None):
        global RR
        return RR[rid].l2_norm(ind=ind)

    def l2_norm(self, ind=None):
        return self.rv.apply(self._l1_norm, self.rid, ind=ind)

    @staticmethod
    def _components(rid, component_indices, ind=None):
        global RR
        return RR[rid].components(component_indices, ind=ind)

    def components(self, component_indices, ind=None):
        return self.rv.apply(self._components, self.rid, component_indices, ind=ind)

    @staticmethod
    def _amax(rid, ind=None):
        global RR
        return RR[rid].amax(ind=ind)

    def amax(self, ind=None):
        return self.rv.apply(self._amax, self.rid, ind=ind)


class RemoteOperator(OperatorInterface):

    dim_source = 0
    dim_range = 0

    type_source = None
    type_range = None

    linear = False

    invert_options = None

    def __init__(self, remote_view, remote_id):
        self.rv = remote_view
        self.rid = remote_id

        @self.rv.remote()
        def get_static_data(rid):
            global RR
            type_source = RR[rid].type_source
            type_range = RR[rid].type_range
            RR[id(type_source)] = type_source
            RR[id(type_range)] = type_range
            return {'type_source': id(type_source),
                    'type_range': id(type_range),
                    'dim_source': RR[rid].dim_source,
                    'dim_range': RR[rid].dim_range,
                    'linear': RR[rid].linear,
                    'invert_options': RR[rid].invert_options,
                    'parameter_type': RR[rid].parameter_type}

        static_data = get_static_data(self.rid)
        pt = static_data.pop('parameter_type')
        self.__dict__.update(get_static_data(self.rid))
        self.type_source = wrap_remote_vector_array(self.rv, self.type_source)
        self.type_range = wrap_remote_vector_array(self.rv, self.type_range)
        self.build_parameter_type(pt, local_global=True)
        self.lock()

    @staticmethod
    def _apply(rid, U, ind=None, mu=None):
        global RR
        U = RR[rid].apply(RR[U], ind=ind, mu=mu)
        U_id = id(U)
        RR[U_id] = U
        return U_id

    def apply(self, U, ind=None, mu=None):
        U_id = self.rv.apply(self._apply, self.rid, U.rid, ind=ind, mu=mu)
        return self.type_range(U_id)

    @staticmethod
    def _apply2(rid, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        global RR
        return RR[rid].apply2(RR[V], RR[U], U_ind=U_ind, V_ind=V_ind, mu=mu, product=product, pairwise=pairwise)

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
        return self.rv.apply(self._apply2, self.rid, V.rid, U.rid, U_ind=U_ind, V_ind=V_ind, mu=mu, product=product,
                             pairwise=pairwise)

    @staticmethod
    def _apply_inverse(rid, U, ind=None, mu=None, options=None):
        global RR
        U = RR[rid].apply_inverse(RR[U], ind=ind, mu=mu, options=options)
        U_id = id(U)
        RR[U_id] = U
        return U_id

    def apply_inverse(self, U, ind=None, mu=None, options=None):
        U_id = self.rv.apply(self._apply_inverse, self.rid, U.rid, ind=ind, mu=mu, options=options)
        return self.type_source(U_id)

    def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        return self.lincomb([self, other], [1, 1])

    __radd__ = __add__

    def __mul__(self, other):
        assert isinstance(other, Number)
        return self.lincomb([self], [other])

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
            self.name, self.dim_source, self.dim_range, self.parameter_type,
            self.__class__.__name__)
