# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# ###########################################################################################
# #
# FIXME This code has not been adapted to the new VectorArray interface and will not work! #
# #
# ###########################################################################################

# flake8: noqa
from __future__ import absolute_import, division, print_function

import os
TRAVIS = os.getenv('TRAVIS') == 'true'

if not TRAVIS:

    from numbers import Number

    import IPython.parallel as p

    from pymor.discretizations.basic import StationaryDiscretization
    from pymor.la.interfaces import VectorArrayInterface
    from pymor.la.numpyvectorarray import NumpyVectorArray
    from pymor.operators.interfaces import OperatorInterface
    from pymor.operators.basic import OperatorBase, ProjectedOperator
    from pymor.operators.constructions import LincombOperator


    class RemoteRessourceManger(object):
        def __init__(self):
            self.refs = {}

        def __setitem__(self, k, v):
            if k in self.refs:
                ref_count, v = self.refs[k]
                self.refs[k] = (ref_count + 1, v)
            else:
                self.refs[k] = (0, v)

        def __getitem__(self, k):
            return self.refs[k][1]

        def __delitem__(self, k):
            ref_count, v = self.refs[k]
            if ref_count == 0:
                del self.refs[k]
            else:
                self.refs[k] = (ref_count - 1, v)


    def setup_remote(remote_view, discretization=None):
        remote_view.block = True
        remote_view.execute('''
    import pymor.playground.remote
    from pymor.operators import LincombOperatorInterface
    pymor.playground.remote.RR = pymor.playground.remote.RemoteRessourceManger()
    ''')
        if discretization:
            remote_view.execute('pymor.playground.remote.RR[id({0})] = {0}'.format(discretization))
            rd = p.Reference(discretization)
            return remote_view.apply(lambda d: id(d), rd)


    wrapped_vector_arrays = {}


    def wrap_remote_vector_array_class(remote_view, remote_id):
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
            return RR[rid].almost_equal(RR[other], ind=ind, o_ind=o_ind, rtol=rtol, atol=atol)

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
            return self.rv.apply(self._l2_norm, self.rid, ind=ind)

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


    def wrap_remote_operator(remote_view, remote_id):
        remote_view.execute('RRES = isinstance(pymor.playground.remote.RR[{}], LincombOperatorInterface)'.format(remote_id))
        if remote_view['RRES']:
            return RemoteLincombOperator(remote_view, remote_id)
        else:
            return RemoteOperator(remote_view, remote_id)


    class RemoteOperator(OperatorInterface):
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
                        'parameter_type': RR[rid].parameter_type,
                        'assemble': hasattr(RR[rid], 'assemble'),
                        'as_vector': hasattr(RR[rid], 'as_vector'),
                        'name': RR[rid].name}

            static_data = get_static_data(self.rid)
            pt = static_data.pop('parameter_type')
            assemble = static_data.pop('assemble')
            as_vector = static_data.pop('as_vector')
            name = static_data.pop('name')
            if assemble:
                self.assemble = self._assemble
            if as_vector:
                self.as_vector = self._as_vector
            self.name = 'Remote_{}'.format(name)
            self.__dict__.update(static_data)
            self.type_source = wrap_remote_vector_array_class(self.rv, self.type_source)
            self.real_type_range = wrap_remote_vector_array_class(self.rv, self.type_range)
            self.type_range = self.real_type_range if self.dim_range > 1 else NumpyVectorArray
            self.build_parameter_type(pt, local_global=True)

        @staticmethod
        def _apply(rid, U, ind=None, mu=None):
            global RR
            U = RR[rid].apply(RR[U], ind=ind, mu=mu)
            U_id = id(U)
            RR[U_id] = U
            return U_id

        def apply(self, U, ind=None, mu=None):
            U_id = self.rv.apply(self._apply, self.rid, U.rid, ind=ind, mu=mu)
            if self.dim_range > 1:
                return self.type_range(U_id)
            else:
                return NumpyVectorArray(self.real_type_range(U_id).components([0]))

        @staticmethod
        def _apply2(rid, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
            global RR
            return RR[rid].apply2(RR[V], RR[U], U_ind=U_ind, V_ind=V_ind, mu=mu, product=product, pairwise=pairwise)

        def apply2(self, V, U, U_ind=None, V_ind=None, mu=None, product=None, pairwise=True):
            if self.dim_range > 1:
                return self.rv.apply(self._apply2, self.rid, V.rid, U.rid, U_ind=U_ind, V_ind=V_ind, mu=mu, product=product,
                                     pairwise=pairwise)
            else:
                assert product is None
                return V.dot(self.apply(U, U_ind, mu=mu), ind=V_ind, pairwise=pairwise)

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

        @staticmethod
        def _lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
            global RR
            op = RR[operators[0]].lincomb([RR[o] for o in operators], coefficients, num_coefficients, coefficients_name,
                                          name)
            op_id = id(op)
            RR[op_id] = op
            return op_id

        @staticmethod
        def lincomb(operators, coefficients=None, num_coefficients=None, coefficients_name=None, name=None):
            assert all(isinstance(op, RemoteOperator) for op in operators)
            op_id = operators[0].rv.apply(operators[0]._lincomb, [o.rid for o in operators], coefficients, num_coefficients,
                                          coefficients_name, name)
            return wrap_remote_operator(operators[0].rv, op_id)

        @staticmethod
        def _s_assemble(rid, mu=None):
            global RR
            op = RR[rid].assemble(mu)
            op_id = id(op)
            RR[op_id] = op
            return op_id

        def _assemble(self, mu=None):
            op_id = self.rv.apply(self._s_assemble, self.rid, mu=mu)
            return wrap_remote_operator(self.rv, op_id)

        @staticmethod
        def _s_as_vector(rid, mu=None):
            global RR
            U = RR[rid].as_vector(mu)
            U_id = id(U)
            RR[U_id] = U
            return U_id

        def _as_vector(self, mu=None):
            U_id = self.rv.apply(self._s_as_vector, self.rid, mu=mu)
            return self.type_source(U_id)

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

        def projected(self, source_basis, range_basis, product=None, name=None):
            name = name or '{}_projected'.format(self.name)
            if self.linear:
                return ProjectedLinearOperator(self, source_basis, range_basis, product, name)
            else:
                return ProjectedOperator(self, source_basis, range_basis, product, name)


    class RemoteLincombOperator(RemoteOperator):
        def __init__(self, remote_view, remote_id):
            RemoteOperator.__init__(self, remote_view, remote_id)
            self.unlock()

            @self.rv.remote()
            def get_static_data(rid):
                global RR
                for op in RR[rid].operators:
                    RR[id(op)] = op
                return {'operators': [id(op) for op in RR[rid].operators],
                        'coefficients': RR[rid].coefficients,
                        'num_coefficients': RR[rid].num_coefficients,
                        'coefficients_name': RR[rid].coefficients_name}

            static_data = get_static_data(self.rid)
            operators = static_data.pop('operators')
            self.__dict__.update(static_data)
            self.operators = [wrap_remote_operator(self.rv, o) for o in operators]

        projected = OperatorBase.projected


    # noinspection PyShadowingNames,PyShadowingNames
    class RemoteStationaryDiscretization(StationaryDiscretization):

        sid_ignore = ('cache_region', 'name')

        def __init__(self, remote_view, remote_id):

            self.rv = remote_view
            self.rid = remote_id

            @self.rv.remote()
            def get_static_data(rid):
                global RR
                self = RR[rid]
                RR[id(self.operator)] = self.operator
                RR[id(self.rhs)] = self.rhs
                for p in self.products.values():
                    RR[id(p)] = p
                return {'operator': id(self.operator),
                        'rhs': id(self.rhs),
                        'products': {k: id(v) for k, v in self.products.iteritems()},
                        'parameter_space': self.parameter_space,
                        'estimator': hasattr(self, 'estimate'),
                        'name': self.name}

            static_data = get_static_data(self.rid)
            super(RemoteStationaryDiscretization, self).__init__(self, operator=wrap_remote_operator(self.rv, static_data[
                'operator']),
                                                                 rhs=wrap_remote_operator(self.rv, static_data['rhs']),
                                                                 products={k: wrap_remote_operator(self.rv, v)
                                                                           for k, v in static_data['products'].iteritems()},
                                                                 parameter_space=static_data['parameter_space'],
                                                                 estimator=None, visualizer=None, cache_region=None,
                                                                 name='Remote_{}'.format(static_data['name']))

            if static_data['estimator']:
                self.unlock()
                self.estimate = self.__estimate


        def with_(self, **kwargs):
            assert set(kwargs.keys()) <= self.with_arguments
            assert 'operators' not in kwargs or 'rhs' not in kwargs and 'operator' not in kwargs
            assert 'operators' not in kwargs or set(kwargs['operators'].keys()) <= {'operator', 'rhs'}

            if 'operators' in kwargs:
                kwargs.update(kwargs.pop('operators'))

            return self._with_via_init(kwargs, new_class=StationaryDiscretization)

        @staticmethod
        def _solve(rid, mu=None):
            global RR
            U = RR[rid].solve(mu)
            U_id = id(U)
            RR[U_id] = U
            return U_id

        def solve(self, mu=None):
            if not self.logging_disabled:
                self.logger.info('Solving {} for {} ...'.format(self.name, mu))
            U_id = self.rv.apply(self._solve, self.rid, mu)
            return self.operator.type_source(U_id)

        @staticmethod
        def _estimate(rid, U, mu=None):
            global RR
            return RR[rid].estimate(RR[U], mu=mu)

        def __estimate(self, U, mu=None):
            return self.rv.apply(self._estimate, self.rid, U.rid, mu=mu)
