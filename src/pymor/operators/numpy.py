# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides the following |NumPy| based |Operators|:

  - |NumpyMatrixOperator| wraps a 2D |NumPy array| as an |Operator|.
  - |NumpyMatrixBasedOperator| should be used as base class for all |Operators|
    which assemble into a |NumpyMatrixOperator|.
  - |NumpyGenericOperator| wraps an arbitrary Python function between
    |NumPy arrays| as an |Operator|.
"""

from functools import reduce

import numpy as np
import scipy.sparse
from scipy.sparse import issparse
from scipy.io import mmwrite, savemat

from pymor.core.config import config
from pymor.core.defaults import defaults, defaults_sid
from pymor.core.exceptions import InversionError
from pymor.core.interfaces import abstractmethod
from pymor.core.logger import getLogger
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NumpyGenericOperator(OperatorBase):
    """Wraps an arbitrary Python function between |NumPy arrays| as a an |Operator|.

    Parameters
    ----------
    mapping
        The function to wrap. If `parameter_type` is `None`, the function is of
        the form `mapping(U)` and is expected to be vectorized. In particular::

            mapping(U).shape == U.shape[:-1] + (dim_range,).

        If `parameter_type` is not `None`, the function has to have the signature
        `mapping(U, mu)`.
    transpose_mapping
        The transpsoe function to wrap. If `parameter_type` is `None`, the function is of
        the form `transpose_mapping(U)` and is expected to be vectorized. In particular::

            transpose_mapping(U).shape == U.shape[:-1] + (dim_source,).

        If `parameter_type` is not `None`, the function has to have the signature
        `transpose_mapping(U, mu)`.
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    linear
        Set to `True` if the provided `mapping` and `transpose_mapping` are linear.
    parameter_type
        The |ParameterType| of the |Parameters| the mapping accepts.
    name
        Name of the operator.
    """

    def __init__(self, mapping, transpose_mapping=None, dim_source=1, dim_range=1, linear=False, parameter_type=None,
                 source_id=None, range_id=None, solver_options=None, name=None):
        self.source = NumpyVectorSpace(dim_source, source_id)
        self.range = NumpyVectorSpace(dim_range, range_id)
        self.solver_options = solver_options
        self.name = name
        self._mapping = mapping
        self._transpose_mapping = transpose_mapping
        self.linear = linear
        if parameter_type is not None:
            self.build_parameter_type(parameter_type)
        self.source_id = source_id  # needed for with_
        self.range_id = range_id

    def apply(self, U, mu=None):
        assert U in self.source
        if self.parametric:
            mu = self.parse_parameter(mu)
            return self.range.make_array(self._mapping(U.data, mu=mu))
        else:
            return self.range.make_array(self._mapping(U.data))

    def apply_transpose(self, V, mu=None):
        if self._transpose_mapping is None:
            raise ValueError('NumpyGenericOperator: transpose mapping was not defined.')
        assert V in self.range
        V = V.data
        if self.parametric:
            mu = self.parse_parameter(mu)
            return self.source.make_array(self._transpose_mapping(V, mu=mu))
        else:
            return self.source.make_array(self._transpose_mapping(V))


class NumpyMatrixBasedOperator(OperatorBase):
    """Base class for operators which assemble into a |NumpyMatrixOperator|.

    Attributes
    ----------
    sparse
        `True` if the operator assembles into a sparse matrix, `False` if the
        operator assembles into a dense matrix, `None` if unknown.
    """

    linear = True
    sparse = None

    @property
    def T(self):
        if not self.parametric:
            return self.assemble().T
        else:
            return super().T

    @abstractmethod
    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        """Assembles the operator for a given |Parameter|.

        Parameters
        ----------
        mu
            The |Parameter| for which to assemble the operator.

        Returns
        -------
        The assembled parameter independent |Operator|.
        """
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid != defaults_sid():
                self.logger.warn('Re-assembling since state of global defaults has changed.')
                op = self._assembled_operator = NumpyMatrixOperator(self._assemble(),
                                                                    source_id=self.source.id,
                                                                    range_id=self.range.id,
                                                                    solver_options=self.solver_options)
                self._defaults_sid = defaults_sid()
                return op
            else:
                return self._assembled_operator
        elif not self.parameter_type:
            op = self._assembled_operator = NumpyMatrixOperator(self._assemble(),
                                                                source_id=self.source.id,
                                                                range_id=self.range.id,
                                                                solver_options=self.solver_options)
            self._defaults_sid = defaults_sid()
            return op
        else:
            return NumpyMatrixOperator(self._assemble(self.parse_parameter(mu)),
                                       source_id=self.source.id,
                                       range_id=self.range.id,
                                       solver_options=self.solver_options)

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_transpose(self, V, mu=None):
        return self.assemble(mu).apply_transpose(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, least_squares=least_squares)

    def export_matrix(self, filename, matrix_name=None, output_format='matlab', mu=None):
        """Save the matrix of the operator to a file.

        Parameters
        ----------
        filename
            Name of output file.
        matrix_name
            The name, the output matrix is given. (Comment field is used in
            case of Matrix Market output_format.) If `None`, the |Operator|'s `name`
            is used.
        output_format
            Output file format. Either `matlab` or `matrixmarket`.
        mu
            The |Parameter| to assemble the to be exported matrix for.
        """
        assert output_format in {'matlab', 'matrixmarket'}
        matrix = self.assemble(mu)._matrix
        matrix_name = matrix_name or self.name
        if output_format is 'matlab':
            savemat(filename, {matrix_name: matrix})
        else:
            mmwrite(filename, matrix, comment=matrix_name)

    def __getstate__(self):
        d = self.__dict__.copy()
        if '_assembled_operator' in d:
            del d['_assembled_operator']
        return d


class NumpyMatrixOperator(NumpyMatrixBasedOperator):
    """Wraps a 2D |NumPy Array| as an |Operator|.

    Parameters
    ----------
    matrix
        The |NumPy array| which is to be wrapped.
    name
        Name of the operator.
    """

    def __init__(self, matrix, source_id=None, range_id=None, solver_options=None, name=None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        self.source = NumpyVectorSpace(matrix.shape[1], source_id)
        self.range = NumpyVectorSpace(matrix.shape[0], range_id)
        self.solver_options = solver_options
        self.name = name
        self._matrix = matrix
        self.source_id = source_id
        self.range_id = range_id
        self.sparse = issparse(matrix)

    @classmethod
    def from_file(cls, path, key=None, source_id=None, range_id=None, solver_options=None, name=None):
        from pymor.tools.io import load_matrix
        matrix = load_matrix(path, key=key)
        return cls(matrix, solver_options=solver_options, source_id=source_id, range_id=range_id,
                   name=name or key or path)

    @property
    def T(self):
        options = {'inverse': self.solver_options.get('inverse_transpose'),
                   'inverse_transpose': self.solver_options.get('inverse')} if self.solver_options else None
        return self.with_(matrix=self._matrix.T, source_id=self.range_id, range_id=self.source_id,
                          solver_options=options, name=self.name + '_transposed')

    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        return self

    def as_range_array(self, mu=None):
        assert not self.sparse
        return self.range.make_array(self._matrix.T.copy())

    def as_source_array(self, mu=None):
        assert not self.sparse
        return self.source.make_array(self._matrix.copy())

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array(self._matrix.dot(U.data.T).T)

    def apply_transpose(self, V, mu=None):
        assert V in self.range
        return self.source.make_array(self._matrix.T.dot(V.data.T).T)

    @defaults('check_finite', 'default_sparse_solver_backend',
              qualname='pymor.operators.numpy.NumpyMatrixOperator.apply_inverse')
    def apply_inverse(self, V, mu=None, least_squares=False, check_finite=True,
                      default_sparse_solver_backend='scipy'):
        """Apply the inverse operator.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the inverse operator is applied.
        mu
            The |Parameter| for which to evaluate the inverse operator.
        least_squares
            If `True`, solve the least squares problem::

                u = argmin ||op(u) - v||_2.

            Since for an invertible operator the least squares solution agrees
            with the result of the application of the inverse operator,
            setting this option should, in general, have no effect on the result
            for those operators. However, note that when no appropriate
            |solver_options| are set for the operator, most implementations
            will choose a least squares solver by default which may be
            undesirable.
        check_finite
            Test if solution only containes finite values.
        default_solver
            Default sparse solver backend to use (scipy, pyamg, generic).

        Returns
        -------
        |VectorArray| of the inverse operator evaluations.

        Raises
        ------
        InversionError
            The operator could not be inverted.
        """
        assert V in self.range

        if V.dim == 0:
            if self.source.dim == 0 or least_squares:
                return self.source.make_array(np.zeros((len(V), self.source.dim)))
            else:
                raise InversionError

        options = self.solver_options.get('inverse') if self.solver_options else None
        assert self.sparse or not options

        if self.sparse:
            if options:
                solver = options if isinstance(options, str) else options['type']
                backend = solver.split('_')[0]
            else:
                backend = default_sparse_solver_backend

            if backend == 'scipy':
                from pymor.bindings.scipy import apply_inverse as apply_inverse_impl
            elif backend == 'pyamg':
                if not config.HAVE_PYAMG:
                    raise RuntimeError('PyAMG support not enabled.')
                from pymor.bindings.pyamg import apply_inverse as apply_inverse_impl
            elif backend == 'generic':
                logger = getLogger('pymor.bindings.scipy.scipy_apply_inverse')
                logger.warn('You have selected a (potentially slow) generic solver for a NumPy matrix operator!')
                from pymor.algorithms.genericsolvers import apply_inverse as apply_inverse_impl
            else:
                raise NotImplementedError

            return apply_inverse_impl(self, V, options=options, least_squares=least_squares, check_finite=check_finite)

        else:
            if least_squares:
                try:
                    R, _, _, _ = np.linalg.lstsq(self._matrix, V.data.T)
                except np.linalg.LinAlgError as e:
                    raise InversionError('{}: {}'.format(str(type(e)), str(e)))
                R = R.T
            else:
                try:
                    R = np.linalg.solve(self._matrix, V.data.T).T
                except np.linalg.LinAlgError as e:
                    raise InversionError('{}: {}'.format(str(type(e)), str(e)))

            if check_finite:
                if not np.isfinite(np.sum(R)):
                    raise InversionError('Result contains non-finite values')

            return self.source.make_array(R)

    def apply_inverse_transpose(self, U, mu=None, least_squares=False):
        options = {'inverse': self.solver_options.get('inverse_transpose') if self.solver_options else None}
        transpose_op = NumpyMatrixOperator(self._matrix.T, source_id=self.range.id, range_id=self.source.id,
                                           solver_options=options)
        return transpose_op.apply_inverse(U, mu=mu, least_squares=least_squares)

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        """Project the operator to a subbasis.

        The purpose of this method is to further project an operator that has been
        obtained through :meth:`~pymor.operators.interfaces.OperatorInterface.projected`
        to subbases of the original projection bases, i.e. ::

            op.projected(r_basis, s_basis, prod).projected_to_subbasis(dim_range, dim_source)

        should be the same as ::

            op.projected(r_basis[:dim_range], s_basis[:dim_source], prod)

        For a |NumpyMatrixOperator| this amounts to extracting the upper-left
        (dim_range, dim_source) corner of the matrix it wraps.

        Parameters
        ----------
        dim_range
            Dimension of the range subbasis.
        dim_source
            Dimension of the source subbasis.
        name
            optional name for the returned |Operator|
        Returns
        -------
        The projected |Operator|.
        """
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        name = name or '{}_projected_to_subbasis'.format(self.name)
        # copy instead of just slicing the matrix to ensure contiguous memory
        return NumpyMatrixOperator(self._matrix[:dim_range, :dim_source].copy(),
                                   source_id=self.source.id,
                                   range_id=self.range.id,
                                   solver_options=self.solver_options,
                                   name=name)

    def assemble_lincomb(self, operators, coefficients, solver_options=None, name=None):
        if not all(isinstance(op, (NumpyMatrixOperator, ZeroOperator, IdentityOperator)) for op in operators):
            return None

        common_mat_dtype = reduce(np.promote_types,
                                  (op._matrix.dtype for op in operators if hasattr(op, '_matrix')))
        common_coef_dtype = reduce(np.promote_types, (type(c.real if c.imag == 0 else c) for c in coefficients))
        common_dtype = np.promote_types(common_mat_dtype, common_coef_dtype)

        if coefficients[0] == 1:
            matrix = operators[0]._matrix.astype(common_dtype)
        else:
            if coefficients[0].imag == 0:
                matrix = operators[0]._matrix * coefficients[0].real
            else:
                matrix = operators[0]._matrix * coefficients[0]
            if matrix.dtype != common_dtype:
                matrix = matrix.astype(common_dtype)

        for op, c in zip(operators[1:], coefficients[1:]):
            if type(op) is ZeroOperator:
                continue
            elif type(op) is IdentityOperator:
                if operators[0].sparse:
                    try:
                        matrix += (scipy.sparse.eye(matrix.shape[0]) * c)
                    except NotImplementedError:
                        matrix = matrix + (scipy.sparse.eye(matrix.shape[0]) * c)
                else:
                    matrix += (np.eye(matrix.shape[0]) * c)
            elif c == 1:
                try:
                    matrix += op._matrix
                except NotImplementedError:
                    matrix = matrix + op._matrix
            elif c == -1:
                try:
                    matrix -= op._matrix
                except NotImplementedError:
                    matrix = matrix - op._matrix
            elif c.imag == 0:
                try:
                    matrix += (op._matrix * c.real)
                except NotImplementedError:
                    matrix = matrix + (op._matrix * c.real)
            else:
                try:
                    matrix += (op._matrix * c)
                except NotImplementedError:
                    matrix = matrix + (op._matrix * c)
        return NumpyMatrixOperator(matrix,
                                   source_id=self.source.id,
                                   range_id=self.range.id,
                                   solver_options=solver_options)

    def __getstate__(self):
        if hasattr(self._matrix, 'factorization'):  # remove unplicklable SuperLU factorization
            del self._matrix.factorization
        return self.__dict__
