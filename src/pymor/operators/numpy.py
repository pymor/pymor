# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
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
from scipy.io import mmwrite, savemat
from scipy.linalg import solve
import scipy.sparse
from scipy.sparse import issparse

from pymor.core.base import abstractmethod
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class NumpyGenericOperator(Operator):
    """Wraps an arbitrary Python function between |NumPy arrays| as an |Operator|.

    Parameters
    ----------
    mapping
        The function to wrap. If `parameter_type` is `None`, the function is of
        the form `mapping(U)` and is expected to be vectorized. In particular::

            mapping(U).shape == U.shape[:-1] + (dim_range,).

        If `parameter_type` is not `None`, the function has to have the signature
        `mapping(U, mu)`.
    adjoint_mapping
        The adjoint function to wrap. If `parameter_type` is `None`, the function is of
        the form `adjoint_mapping(U)` and is expected to be vectorized. In particular::

            adjoint_mapping(U).shape == U.shape[:-1] + (dim_source,).

        If `parameter_type` is not `None`, the function has to have the signature
        `adjoint_mapping(U, mu)`.
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    linear
        Set to `True` if the provided `mapping` and `adjoint_mapping` are linear.
    parameter_type
        The |ParameterType| of the |Parameters| the mapping accepts.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, mapping, adjoint_mapping=None, dim_source=1, dim_range=1, linear=False, parameter_type=None,
                 source_id=None, range_id=None, solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(dim_source, source_id)
        self.range = NumpyVectorSpace(dim_range, range_id)

    def apply(self, U, mu=None):
        assert U in self.source
        if self.parametric:
            mu = self.parse_parameter(mu)
            return self.range.make_array(self.mapping(U.to_numpy(), mu=mu))
        else:
            return self.range.make_array(self.mapping(U.to_numpy()))

    def apply_adjoint(self, V, mu=None):
        if self.adjoint_mapping is None:
            raise ValueError('NumpyGenericOperator: adjoint mapping was not defined.')
        assert V in self.range
        V = V.to_numpy()
        if self.parametric:
            mu = self.parse_parameter(mu)
            return self.source.make_array(self.adjoint_mapping(V, mu=mu))
        else:
            return self.source.make_array(self.adjoint_mapping(V))


class NumpyMatrixBasedOperator(Operator):
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
    def H(self):
        if not self.parametric:
            return self.assemble().H
        else:
            return super().H

    @abstractmethod
    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        return NumpyMatrixOperator(self._assemble(self.parse_parameter(mu)),
                                   source_id=self.source.id,
                                   range_id=self.range.id,
                                   solver_options=self.solver_options,
                                   name=self.name)

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

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
        matrix = self.assemble(mu).matrix
        matrix_name = matrix_name or self.name
        if output_format == 'matlab':
            savemat(filename, {matrix_name: matrix})
        else:
            mmwrite(filename, matrix, comment=matrix_name)


class NumpyMatrixOperator(NumpyMatrixBasedOperator):
    """Wraps a 2D |NumPy Array| as an |Operator|.

    Parameters
    ----------
    matrix
        The |NumPy array| which is to be wrapped.
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, matrix, source_id=None, range_id=None, solver_options=None, name=None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        try:
            matrix.setflags(write=False)  # make numpy arrays read-only
        except AttributeError:
            pass

        self.__auto_init(locals())
        self.source = NumpyVectorSpace(matrix.shape[1], source_id)
        self.range = NumpyVectorSpace(matrix.shape[0], range_id)
        self.sparse = issparse(matrix)

    @classmethod
    def from_file(cls, path, key=None, source_id=None, range_id=None, solver_options=None, name=None):
        from pymor.tools.io import load_matrix
        matrix = load_matrix(path, key=key)
        return cls(matrix, solver_options=solver_options, source_id=source_id, range_id=range_id,
                   name=name or key or path)

    @property
    def H(self):
        options = {'inverse': self.solver_options.get('inverse_adjoint'),
                   'inverse_adjoint': self.solver_options.get('inverse')} if self.solver_options else None
        if self.sparse:
            adjoint_matrix = self.matrix.transpose(copy=False).conj(copy=False)
        elif np.isrealobj(self.matrix):
            adjoint_matrix = self.matrix.T
        else:
            adjoint_matrix = self.matrix.T.conj()
        return self.with_(matrix=adjoint_matrix, source_id=self.range_id, range_id=self.source_id,
                          solver_options=options, name=self.name + '_adjoint')

    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        return self

    def as_range_array(self, mu=None):
        return self.range.make_array(self.matrix.T.copy())

    def as_source_array(self, mu=None):
        return self.source.make_array(self.matrix.copy()).conj()

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array(self.matrix.dot(U.to_numpy().T).T)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @defaults('check_finite', 'default_sparse_solver_backend')
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
            Test if solution only contains finite values.
        default_sparse_solver_backend
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

        if self.source.dim != self.range.dim and not least_squares:
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
                logger.warning('You have selected a (potentially slow) generic solver for a NumPy matrix operator!')
                from pymor.algorithms.genericsolvers import apply_inverse as apply_inverse_impl
            else:
                raise NotImplementedError

            return apply_inverse_impl(self, V, options=options, least_squares=least_squares, check_finite=check_finite)

        else:
            if least_squares:
                try:
                    R, _, _, _ = np.linalg.lstsq(self.matrix, V.to_numpy().T)
                except np.linalg.LinAlgError as e:
                    raise InversionError(f'{str(type(e))}: {str(e)}')
                R = R.T
            else:
                try:
                    R = solve(self.matrix, V.to_numpy().T).T
                except np.linalg.LinAlgError as e:
                    raise InversionError(f'{str(type(e))}: {str(e)}')

            if check_finite:
                if not np.isfinite(np.sum(R)):
                    raise InversionError('Result contains non-finite values')

            return self.source.make_array(R)

    def apply_inverse_adjoint(self, U, mu=None, least_squares=False):
        return self.H.apply_inverse(U, mu=mu, least_squares=least_squares)

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
        if not all(isinstance(op, NumpyMatrixOperator) for op in operators):
            return None

        common_mat_dtype = reduce(np.promote_types,
                                  (op.matrix.dtype for op in operators if hasattr(op, 'matrix')))
        common_coef_dtype = reduce(np.promote_types, (type(c) for c in coefficients + [identity_shift]))
        common_dtype = np.promote_types(common_mat_dtype, common_coef_dtype)

        if coefficients[0] == 1:
            matrix = operators[0].matrix.astype(common_dtype)
        else:
            matrix = operators[0].matrix * coefficients[0]
            if matrix.dtype != common_dtype:
                matrix = matrix.astype(common_dtype)

        for op, c in zip(operators[1:], coefficients[1:]):
            if c == 1:
                try:
                    matrix += op.matrix
                except NotImplementedError:
                    matrix = matrix + op.matrix
            elif c == -1:
                try:
                    matrix -= op.matrix
                except NotImplementedError:
                    matrix = matrix - op.matrix
            else:
                try:
                    matrix += (op.matrix * c)
                except NotImplementedError:
                    matrix = matrix + (op.matrix * c)

        if identity_shift != 0:
            if identity_shift.imag == 0:
                identity_shift = identity_shift.real
            if operators[0].sparse:
                try:
                    matrix += (scipy.sparse.eye(matrix.shape[0]) * identity_shift)
                except NotImplementedError:
                    matrix = matrix + (scipy.sparse.eye(matrix.shape[0]) * identity_shift)
            else:
                matrix += (np.eye(matrix.shape[0]) * identity_shift)

        return NumpyMatrixOperator(matrix,
                                   source_id=self.source.id,
                                   range_id=self.range.id,
                                   solver_options=solver_options)

    def __getstate__(self):
        if hasattr(self.matrix, 'factorization'):  # remove unplicklable SuperLU factorization
            del self.matrix.factorization
        return self.__dict__

    def _format_repr(self, max_width, verbosity):
        if self.sparse:
            matrix_repr = f'<{self.range.dim}x{self.source.dim} sparse, {self.matrix.nnz} nnz>'
        else:
            matrix_repr = f'<{self.range.dim}x{self.source.dim} dense>'
        return super()._format_repr(max_width, verbosity, override={'matrix': matrix_repr})
