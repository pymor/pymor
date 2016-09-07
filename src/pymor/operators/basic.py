# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from numbers import Number

import numpy as np

from pymor.algorithms import genericsolvers
from pymor.core.exceptions import InversionError
from pymor.operators.interfaces import OperatorInterface
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace


class OperatorBase(OperatorInterface):
    """Base class for |Operators| providing some default implementations.

    When implementing a new operator, it is usually advisable to derive
    from this class.
    """

    def apply2(self, V, U, U_ind=None, V_ind=None, mu=None):
        mu = self.parse_parameter(mu)
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        U_ind = None if U_ind is None else np.array(U_ind, copy=False, dtype=np.int, ndmin=1)
        V_ind = None if V_ind is None else np.array(V_ind, copy=False, dtype=np.int, ndmin=1)
        AU = self.apply(U, ind=U_ind, mu=mu)
        return V.dot(AU, ind=V_ind)

    def pairwise_apply2(self, V, U, U_ind=None, V_ind=None, mu=None):
        mu = self.parse_parameter(mu)
        assert isinstance(V, VectorArrayInterface)
        assert isinstance(U, VectorArrayInterface)
        U_ind = None if U_ind is None else np.array(U_ind, copy=False, dtype=np.int, ndmin=1)
        V_ind = None if V_ind is None else np.array(V_ind, copy=False, dtype=np.int, ndmin=1)
        lu = len(U_ind) if U_ind is not None else len(U)
        lv = len(V_ind) if V_ind is not None else len(V)
        assert lu == lv
        AU = self.apply(U, ind=U_ind, mu=mu)
        return V.pairwise_dot(AU, ind=V_ind)

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
        if isinstance(other, Number):
            assert other == 0.
            return self
        from pymor.operators.constructions import LincombOperator
        return LincombOperator([self, other], [1, -1])

    def __add__(self, other):
        if isinstance(other, Number):
            assert other == 0.
            return self
        from pymor.operators.constructions import LincombOperator
        return LincombOperator([self, other], [1, 1])

    __radd__ = __add__

    def __mul__(self, other):
        assert isinstance(other, Number)
        from pymor.operators.constructions import LincombOperator
        return LincombOperator([self], [other])

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
            self.name, self.source.dim, self.range.dim, self.parameter_type,
            self.__class__.__name__)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        if self.linear:
            raise NotImplementedError
        else:
            raise ValueError('Trying to apply adjoint of nonlinear operator.')

    def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
        from pymor.operators.constructions import FixedParameterOperator
        assembled_op = self.assemble(mu)
        if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
            return assembled_op.apply_inverse(V, ind=ind, least_squares=least_squares)
        elif self.linear:
            options = (self.solver_options.get('inverse') if self.solver_options else
                       'least_squares' if least_squares else
                       None)

            if options and not least_squares:
                solver_type = options if isinstance(options, str) else options['type']
                if solver_type.startswith('least_squares'):
                    self.logger.warn('Least squares solver selected but "least_squares == False"')

            try:
                return genericsolvers.apply_inverse(assembled_op, V.copy(ind), options=options)
            except InversionError as e:
                if least_squares and options:
                    solver_type = options if isinstance(options, str) else options['type']
                    if not solver_type.startswith('least_squares'):
                        msg = str(e) \
                            + '\nNote: linear solver was selected for solving least squares problem ' \
                            + '(maybe not invertible?)'
                        raise InversionError(msg)
                raise e
        else:
            from pymor.algorithms.newton import newton
            from pymor.core.exceptions import NewtonError
            assert V.check_ind(ind)

            options = self.solver_options
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

            ind = (list(range(len(V))) if ind is None else
                   [ind] if isinstance(ind, Number) else
                   ind)
            R = V.empty(reserve=len(ind))
            for i in ind:
                try:
                    R.append(newton(self, V.copy(i), **options)[0])
                except NewtonError as e:
                    raise InversionError(e)
            return R

    def apply_inverse_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None,
                              least_squares=False):
        from pymor.operators.constructions import FixedParameterOperator
        assembled_op = self.assemble(mu)
        if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
            return assembled_op.apply_inverse_adjoint(U, ind=ind, source_product=source_product,
                                                      range_product=range_product, least_squares=least_squares)
        elif source_product or range_product:
            if source_product:
                U = source_product.apply(U, ind=ind)
                ind = None
            # maybe there is a better implementation for source_product == None and range_product == None
            V = self.apply_inverse_adjoint(U, mu=mu, least_squares=least_squares)
            if range_product:
                return range_product.apply_inverse(V)
            else:
                return V
        else:
            if not self.linear:
                raise NotImplementedError
            # use generic solver for the adjoint operator
            from pymor.operators.constructions import AdjointOperator
            options = {'inverse': self.solver_options.get('inverse_adjoint') if self.solver_options else None}
            adjoint_op = AdjointOperator(self, with_apply_inverse=False, solver_options=options)
            return adjoint_op.apply_inverse(U, ind=ind, mu=mu, least_squares=least_squares)

    def as_vector(self, mu=None):
        if not self.linear:
            raise TypeError('This nonlinear operator does not represent a vector or linear functional.')
        elif self.source.dim == 1 and self.source.type is NumpyVectorArray:
            return self.apply(NumpyVectorArray(1), mu=mu)
        elif self.range.dim == 1 and self.range.type is NumpyVectorArray:
            return self.apply_adjoint(NumpyVectorArray(1), mu=mu)
        else:
            raise TypeError('This operator does not represent a vector or linear functional.')

    def projected(self, range_basis, source_basis, product=None, name=None):
        name = name or '{}_projected'.format(self.name)
        if self.linear and not self.parametric:
            assert source_basis is None or source_basis in self.source
            assert range_basis is None or range_basis in self.range
            assert product is None or product.source == product.range == self.range
            if source_basis is None:
                if range_basis is None:
                    return self
                else:
                    try:
                        V = self.apply_adjoint(range_basis, range_product=product)
                    except NotImplementedError:
                        return ProjectedOperator(self, range_basis, None, product, name=name)
                    if self.source.type == NumpyVectorArray:
                        from pymor.operators.numpy import NumpyMatrixOperator
                        return NumpyMatrixOperator(V.data, name=name)
                    else:
                        from pymor.operators.constructions import VectorArrayOperator
                        return VectorArrayOperator(V, transposed=True, name=name)
            else:
                if range_basis is None:
                    V = self.apply(source_basis)
                    if self.range.type == NumpyVectorArray:
                        from pymor.operators.numpy import NumpyMatrixOperator
                        return NumpyMatrixOperator(V.data.T, name=name)
                    else:
                        from pymor.operators.constructions import VectorArrayOperator
                        return VectorArrayOperator(V, transposed=False, name=name)
                elif product is None:
                    from pymor.operators.numpy import NumpyMatrixOperator
                    return NumpyMatrixOperator(self.apply2(range_basis, source_basis), name=name)
                else:
                    from pymor.operators.numpy import NumpyMatrixOperator
                    V = self.apply(source_basis)
                    return NumpyMatrixOperator(product.apply2(range_basis, V), name=name)
        else:
            self.logger.warn('Using inefficient generic projection operator')
            return ProjectedOperator(self, range_basis, source_basis, product, name=name)


class ProjectedOperator(OperatorBase):
    """Generic |Operator| representing the projection of an |Operator| to a subspace.

    This operator is implemented as the concatenation of the linear combination with
    `source_basis`, application of the original `operator` and projection onto
    `range_basis`. As such, this operator can be used to obtain a reduced basis
    projection of any given |Operator|. However, no offline/online decomposition is
    performed, so this operator is mainly useful for testing before implementing
    offline/online decomposition for a specific application.

    This operator is instantiated in the default implementation of
    :meth:`~pymor.operators.interfaces.OperatorInterface.projected` in
    :class:`OperatorBase` for parametric or nonlinear operators.

    Parameters
    ----------
    operator
        The |Operator| to project.
    source_basis
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    range_basis
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    product
        See :meth:`~pymor.operators.interfaces.OperatorInterface.projected`.
    name
        Name of the projected operator.
    """

    linear = False

    def __init__(self, operator, range_basis, source_basis, product=None, solver_options=None, name=None):
        assert isinstance(operator, OperatorInterface)
        assert source_basis is None or source_basis in operator.source
        assert range_basis is None or range_basis in operator.range
        assert product is None \
            or (isinstance(product, OperatorInterface)
                and range_basis is not None
                and operator.range == product.source
                and product.range == product.source)
        self.build_parameter_type(inherits=(operator,))
        self.source = NumpyVectorSpace(len(source_basis)) if source_basis is not None else operator.source
        self.range = NumpyVectorSpace(len(range_basis)) if range_basis is not None else operator.range
        self.solver_options = solver_options
        self.name = name
        self.operator = operator
        self.source_basis = source_basis.copy() if source_basis is not None else None
        self.range_basis = range_basis.copy() if range_basis is not None else None
        self.linear = operator.linear
        self.product = product

    def apply(self, U, ind=None, mu=None):
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            if self.range_basis is None:
                return self.operator.apply(U, ind=ind, mu=mu)
            elif self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, U, U_ind=ind, mu=mu).T)
            else:
                V = self.operator.apply(U, ind=ind, mu=mu)
                return NumpyVectorArray(self.product.apply2(V, self.range_basis))
        else:
            U_array = U._array[:U._len] if ind is None else U._array[ind]
            UU = self.source_basis.lincomb(U_array)
            if self.range_basis is None:
                return self.operator.apply(UU, mu=mu)
            elif self.product is None:
                return NumpyVectorArray(self.operator.apply2(self.range_basis, UU, mu=mu).T)
            else:
                V = self.operator.apply(UU, mu=mu)
                return NumpyVectorArray(self.product.apply2(V, self.range_basis))

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        """See :meth:`NumpyMatrixOperator.projected_to_subbasis`."""
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        assert dim_source is None or self.source_basis is not None, 'not implemented'
        assert dim_range is None or self.range_basis is not None, 'not implemented'
        name = name or '{}_projected_to_subbasis'.format(self.name)
        source_basis = self.source_basis if dim_source is None \
            else self.source_basis.copy(ind=list(range(dim_source)))
        range_basis = self.range_basis if dim_range is None \
            else self.range_basis.copy(ind=list(range(dim_range)))
        return ProjectedOperator(self.operator, range_basis, source_basis, product=None,
                                 solver_options=self.solver_options, name=name)

    def jacobian(self, U, mu=None):
        if self.linear:
            return self.assemble(mu)
        assert len(U) == 1
        mu = self.parse_parameter(mu)
        if self.source_basis is None:
            J = self.operator.jacobian(U, mu=mu)
        else:
            J = self.operator.jacobian(self.source_basis.lincomb(U.data), mu=mu)
        pop = J.projected(range_basis=self.range_basis, source_basis=self.source_basis,
                          product=self.product, name=self.name + '_jacobian')
        if self.solver_options:
            options = self.solver_options.get('jacobian')
            if options:
                pop = pop.with_(solver_options=options)
        return pop

    def assemble(self, mu=None):
        op = self.operator.assemble(mu=mu)
        pop = op.projected(range_basis=self.range_basis, source_basis=self.source_basis,
                           product=self.product, name=self.name + '_assembled')
        if self.solver_options:
            pop = pop.with_(solver_options=self.solver_options)
        return pop
