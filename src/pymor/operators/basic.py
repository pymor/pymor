# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms import genericsolvers
from pymor.core.exceptions import InversionError, LinAlgError
from pymor.core.interfaces import abstractmethod
from pymor.operators.interfaces import OperatorInterface
from pymor.parameters.interfaces import ParameterFunctionalInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorSpace


class OperatorBase(OperatorInterface):
    """Base class for |Operators| providing some default implementations.

    When implementing a new operator, it is usually advisable to derive
    from this class.
    """

    def apply2(self, V, U, mu=None):
        mu = self.parse_parameter(mu)
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        AU = self.apply(U, mu=mu)
        return V.dot(AU)

    def pairwise_apply2(self, V, U, mu=None):
        mu = self.parse_parameter(mu)
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        assert len(U) == len(V)
        AU = self.apply(U, mu=mu)
        return V.pairwise_dot(AU)

    def jacobian(self, U, mu=None):
        if self.linear:
            if self.parametric:
                return self.assemble(mu)
            else:
                return self
        else:
            raise NotImplementedError

    def assemble(self, mu=None):
        if self.parametric:
            from pymor.operators.constructions import FixedParameterOperator

            return FixedParameterOperator(self, mu=mu, name=self.name + '_assembled')
        else:
            return self

    def __sub__(self, other):
        if not isinstance(other, OperatorInterface):
            return NotImplemented
        from pymor.operators.constructions import LincombOperator
        if isinstance(other, LincombOperator):
            return NotImplemented
        else:
            return LincombOperator([self, other], [1., -1.])

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, OperatorInterface):
            return NotImplemented
        from pymor.operators.constructions import LincombOperator
        if isinstance(other, LincombOperator):
            return NotImplemented
        else:
            return LincombOperator([self, other], [1., 1.])

    __radd__ = __add__

    def __mul__(self, other):
        if not isinstance(other, (Number, ParameterFunctionalInterface)):
            return NotImplemented
        from pymor.operators.constructions import LincombOperator
        return LincombOperator([self], [other])

    def __matmul__(self, other):
        if not isinstance(other, OperatorInterface):
            return NotImplemented
        from pymor.operators.constructions import Concatenation
        if isinstance(other, Concatenation):
            return NotImplemented
        else:
            return Concatenation((self, other))

    def __str__(self):
        return f'{self.name}: R^{self.source.dim} --> R^{self.range.dim}  ' \
               f'(parameter type: {self.parameter_type}, class: {self.__class__.__name__})'

    def apply_adjoint(self, V, mu=None):
        if self.linear:
            raise NotImplementedError
        else:
            raise LinAlgError('Operator not linear.')

    def apply_inverse(self, V, mu=None, least_squares=False):
        from pymor.operators.constructions import FixedParameterOperator
        assembled_op = self.assemble(mu)
        if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
            return assembled_op.apply_inverse(V, least_squares=least_squares)
        elif self.linear:
            options = self.solver_options.get('inverse') if self.solver_options else None
            return genericsolvers.apply_inverse(assembled_op, V, options=options, least_squares=least_squares)
        else:
            from pymor.algorithms.newton import newton
            from pymor.core.exceptions import NewtonError

            options = self.solver_options.get('inverse') if self.solver_options else None
            if options:
                if isinstance(options, str):
                    assert options == 'newton'
                    options = {}
                else:
                    assert options['type'] == 'newton'
                    options = options.copy()
                    options.pop('type')
            else:
                options = {}
            options['least_squares'] = least_squares

            R = V.empty(reserve=len(V))
            for i in range(len(V)):
                try:
                    R.append(newton(self, V[i], mu=mu, **options)[0])
                except NewtonError as e:
                    raise InversionError(e)
            return R

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        from pymor.operators.constructions import FixedParameterOperator
        if not self.linear:
            raise LinAlgError('Operator not linear.')
        assembled_op = self.assemble(mu)
        if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
            return assembled_op.apply_inverse_adjoint(U, least_squares=least_squares)
        else:
            # use generic solver for the adjoint operator
            from pymor.operators.constructions import AdjointOperator
            options = {'inverse': self.solver_options.get('inverse_adjoint') if self.solver_options else None}
            adjoint_op = AdjointOperator(self, with_apply_inverse=False, solver_options=options)
            return adjoint_op.apply_inverse(U, mu=mu, least_squares=least_squares)

    def as_range_array(self, mu=None):
        return self.apply(self.source.from_numpy(np.eye(self.source.dim)), mu=mu)

    def as_source_array(self, mu=None):
        return self.apply_adjoint(self.range.from_numpy(np.eye(self.range.dim)), mu=mu)

    def d_mu(self, component, index=()):
        if self.parametric:
            raise NotImplementedError
        else:
            from pymor.operators.constructions import ZeroOperator
            return ZeroOperator(self.range, self.source, name=self.name + '_d_mu')

class ListVectorArrayOperatorBase(OperatorBase):

    def _prepare_apply(self, U, mu, kind, least_squares=False):
        pass

    @abstractmethod
    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def apply(self, U, mu=None):
        assert U in self.source
        data = self._prepare_apply(U, mu, 'apply')
        V = [self._apply_one_vector(u, mu=mu, prepare_data=data) for u in U._list]
        return self.range.make_array(V)

    def apply_inverse(self, V, mu=None, least_squares=False):
        assert V in self.range
        try:
            data = self._prepare_apply(V, mu, 'apply_inverse', least_squares=least_squares)
            U = [self._apply_inverse_one_vector(v, mu=mu, least_squares=least_squares, prepare_data=data)
                 for v in V._list]
        except NotImplementedError:
            return super().apply_inverse(V, mu=mu, least_squares=least_squares)
        return self.source.make_array(U)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        try:
            data = self._prepare_apply(V, mu, 'apply_adjoint')
            U = [self._apply_adjoint_one_vector(v, mu=mu, prepare_data=data) for v in V._list]
        except NotImplementedError:
            return super().apply_adjoint(V, mu=mu)
        return self.source.make_array(U)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        assert U in self.source
        try:
            data = self._prepare_apply(U, mu, 'apply_inverse_adjoint', least_squares=least_squares)
            V = [self._apply_inverse_adjoint_one_vector(u, mu=mu, least_squares=least_squares, prepare_data=data)
                 for u in U._list]
        except NotImplementedError:
            return super().apply_inverse_adjoint(U, mu=mu, least_squares=least_squares)
        return self.range.make_array(V)


class LinearComplexifiedListVectorArrayOperatorBase(ListVectorArrayOperatorBase):

    linear = True

    @abstractmethod
    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        pass

    def _real_apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        raise NotImplementedError

    def _real_apply_inverse_adjoint_one_vector(self, u, mu=None, least_squares=False, prepare_data=None):
        raise NotImplementedError

    def _apply_one_vector(self, u, mu=None, prepare_data=None):
        real_part = self._real_apply_one_vector(u.real_part, mu=mu, prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_one_vector(u.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.complexified_vector_type(real_part, imag_part)

    def _apply_inverse_one_vector(self, v, mu=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_one_vector(v.real_part, mu=mu, least_squares=least_squares,
                                                        prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_inverse_one_vector(v.imag_part, mu=mu, least_squares=least_squares,
                                                            prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.complexified_vector_type(real_part, imag_part)

    def _apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        real_part = self._real_apply_adjoint_one_vector(v.real_part, mu=mu, prepare_data=prepare_data)
        if v.imag_part is not None:
            imag_part = self._real_apply_adjoint_one_vector(v.imag_part, mu=mu, prepare_data=prepare_data)
        else:
            imag_part = None
        return self.source.complexified_vector_type(real_part, imag_part)

    def _apply_inverse_adjoint_one_vector(self, u, mu=None, least_squares=False, prepare_data=None):
        real_part = self._real_apply_inverse_adjoint_one_vector(u.real_part, mu=mu, least_squares=least_squares,
                                                                prepare_data=prepare_data)
        if u.imag_part is not None:
            imag_part = self._real_apply_inverse_adjoint_one_vector(u.imag_part, mu=mu, least_squares=least_squares,
                                                                    prepare_data=prepare_data)
        else:
            imag_part = None
        return self.range.complexified_vector_type(real_part, imag_part)


class ProjectedOperator(OperatorBase):
    """Generic |Operator| representing the projection of an |Operator| to a subspace.

    This operator is implemented as the concatenation of the linear combination with
    `source_basis`, application of the original `operator` and projection onto
    `range_basis`. As such, this operator can be used to obtain a reduced basis
    projection of any given |Operator|. However, no offline/online decomposition is
    performed, so this operator is mainly useful for testing before implementing
    offline/online decomposition for a specific application.

    This operator is instantiated in :func:`pymor.algorithms.projection.project`
    as a default implementation for parametric or nonlinear operators.

    Parameters
    ----------
    operator
        The |Operator| to project.
    range_basis
        See :func:`pymor.algorithms.projection.project`.
    source_basis
        See :func:`pymor.algorithms.projection.project`.
    product
        See :func:`pymor.algorithms.projection.project`.
    solver_options
        The |solver_options| for the projected operator.
    """

    linear = False

    def __init__(self, operator, range_basis, source_basis, product=None, solver_options=None):
        assert isinstance(operator, OperatorInterface)
        assert source_basis is None or source_basis in operator.source
        assert range_basis is None or range_basis in operator.range
        assert (product is None
                or (isinstance(product, OperatorInterface)
                    and range_basis is not None
                    and operator.range == product.source
                    and product.range == product.source))
        if source_basis is not None:
            source_basis = source_basis.copy()
        if range_basis is not None:
            range_basis = range_basis.copy()
        self.__auto_init(locals())
        self.build_parameter_type(operator)
        self.source = NumpyVectorSpace(len(source_basis)) if source_basis is not None else operator.source
        self.range = NumpyVectorSpace(len(range_basis)) if range_basis is not None else operator.range
        self.linear = operator.linear

    @property
    def H(self):
        if self.product:
            return super().H
        else:
            options = {'inverse': self.solver_options.get('inverse_adjoint'),
                       'inverse_adjoint': self.solver_options.get('inverse')} if self.solver_options else None
            return ProjectedOperator(self.operator.H, self.source_basis, self.range_basis, solver_options=options)

    def apply(self, U, mu=None):
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            if self.range_basis is None:
                return self.operator.apply(U, mu=mu)
            elif self.product is None:
                return self.range.make_array(self.operator.apply2(self.range_basis, U, mu=mu).T)
            else:
                V = self.operator.apply(U, mu=mu)
                return self.range.make_array(self.product.apply2(V, self.range_basis))
        else:
            UU = self.source_basis.lincomb(U.to_numpy())
            if self.range_basis is None:
                return self.operator.apply(UU, mu=mu)
            elif self.product is None:
                return self.range.make_array(self.operator.apply2(self.range_basis, UU, mu=mu).T)
            else:
                V = self.operator.apply(UU, mu=mu)
                return self.range.make_array(self.product.apply2(V, self.range_basis))

    def jacobian(self, U, mu=None):
        if self.linear:
            return self.assemble(mu)
        assert len(U) == 1
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            J = self.operator.jacobian(U, mu=mu)
        else:
            J = self.operator.jacobian(self.source_basis.lincomb(U.to_numpy()), mu=mu)
        from pymor.algorithms.projection import project
        pop = project(J, range_basis=self.range_basis, source_basis=self.source_basis,
                      product=self.product)
        if self.solver_options:
            options = self.solver_options.get('jacobian')
            if options:
                pop = pop.with_(solver_options=options)
        return pop

    def assemble(self, mu=None):
        op = self.operator.assemble(mu=mu)
        if op == self.operator:  # avoid infinite recursion in apply_inverse default impl
            return self
        from pymor.algorithms.projection import project
        pop = project(op, range_basis=self.range_basis, source_basis=self.source_basis,
                      product=self.product)
        if self.solver_options:
            pop = pop.with_(solver_options=self.solver_options)
        return pop

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        if self.range_basis is not None:
            V = self.range_basis.lincomb(V.to_numpy())
        U = self.operator.apply_adjoint(V, mu)
        if self.source_basis is not None:
            U = self.source.make_array(U.dot(self.source_basis))
        return U
