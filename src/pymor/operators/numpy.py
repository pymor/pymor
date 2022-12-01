# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""|Operators| based on |NumPy| arrays.

This module provides the following |NumPy|-based |Operators|:

- |NumpyMatrixOperator| wraps a 2D |NumPy array| as an |Operator|.
- |NumpyMatrixBasedOperator| should be used as base class for all |Operators|
  which assemble into a |NumpyMatrixOperator|.
- |NumpyGenericOperator| wraps an arbitrary Python function between
  |NumPy arrays| as an |Operator|.
- |NumpyHankelOperator| implicitly constructs a Hankel operator from a |NumPy array| of
  Markov parameters.
"""

from functools import reduce

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft
from scipy.io import mmwrite, savemat
from scipy.linalg import solve
import scipy.sparse
from scipy.sparse import issparse

from pymor.core.base import abstractmethod
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
        The function to wrap. If `parameters` is `None`, the function is of
        the form `mapping(U)` and is expected to be vectorized. In particular::

            mapping(U).shape == U.shape[:-1] + (dim_range,).

        If `parameters` is not `None`, the function has to have the signature
        `mapping(U, mu)`.
    adjoint_mapping
        The adjoint function to wrap. If `parameters` is `None`, the function is of
        the form `adjoint_mapping(U)` and is expected to be vectorized. In particular::

            adjoint_mapping(U).shape == U.shape[:-1] + (dim_source,).

        If `parameters` is not `None`, the function has to have the signature
        `adjoint_mapping(U, mu)`.
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    linear
        Set to `True` if the provided `mapping` and `adjoint_mapping` are linear.
    parameters
        The |Parameters| the operator depends on.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, mapping, adjoint_mapping=None, dim_source=1, dim_range=1, linear=False, parameters={},
                 source_id=None, range_id=None, solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(dim_source, source_id)
        self.range = NumpyVectorSpace(dim_range, range_id)
        self.parameters_own = parameters

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        if self.parametric:
            return self.range.make_array(self.mapping(U.to_numpy(), mu=mu))
        else:
            return self.range.make_array(self.mapping(U.to_numpy()))

    def apply_adjoint(self, V, mu=None):
        if self.adjoint_mapping is None:
            raise ValueError('NumpyGenericOperator: adjoint mapping was not defined.')
        assert V in self.range
        assert self.parameters.assert_compatible(mu)
        V = V.to_numpy()
        if self.parametric:
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
        assert self.parameters.assert_compatible(mu)
        return NumpyMatrixOperator(self._assemble(mu),
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

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)

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
            The |parameter values| to assemble the to be exported matrix for.
        """
        assert output_format in {'matlab', 'matrixmarket'}
        matrix = self.assemble(mu).matrix
        matrix_name = matrix_name or self.name
        if output_format == 'matlab':
            savemat(filename, {matrix_name: matrix})
        else:
            mmwrite(filename, matrix, comment=matrix_name)


class NumpyMatrixOperator(NumpyMatrixBasedOperator):
    """Wraps a 2D |NumPy Array| or |SciPy spmatrix| as an |Operator|.

    Parameters
    ----------
    matrix
        The |NumPy array| or |SciPy spmatrix| which is to be wrapped.
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
        if self.sparse:
            return Operator.as_range_array(self)
        return self.range.from_numpy(self.matrix.T.copy())

    def as_source_array(self, mu=None):
        if self.sparse:
            return Operator.as_source_array(self)
        return self.source.from_numpy(self.matrix.copy()).conj()

    def apply(self, U, mu=None):
        assert U in self.source
        return self.range.make_array(self.matrix.dot(U.to_numpy().T).T)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @defaults('check_finite', 'default_sparse_solver_backend')
    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False,
                      check_finite=True, default_sparse_solver_backend='scipy'):
        """Apply the inverse operator.

        Parameters
        ----------
        V
            |VectorArray| of vectors to which the inverse operator is applied.
        mu
            The |parameter values| for which to evaluate the inverse operator.
        initial_guess
            |VectorArray| with the same length as `V` containing initial guesses
            for the solution.  Some implementations of `apply_inverse` may
            ignore this parameter.  If `None` a solver-dependent default is used.
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
            Default sparse solver backend to use (scipy, generic).

        Returns
        -------
        |VectorArray| of the inverse operator evaluations.

        Raises
        ------
        InversionError
            The operator could not be inverted.
        """
        assert V in self.range
        assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)

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
            elif backend == 'generic':
                logger = getLogger('pymor.bindings.scipy.scipy_apply_inverse')
                logger.warning('You have selected a (potentially slow) generic solver for a NumPy matrix operator!')
                from pymor.algorithms.genericsolvers import apply_inverse as apply_inverse_impl
            else:
                raise NotImplementedError

            return apply_inverse_impl(self, V, initial_guess=initial_guess, options=options,
                                      least_squares=least_squares, check_finite=check_finite)

        else:
            if least_squares:
                try:
                    R, _, _, _ = np.linalg.lstsq(self.matrix, V.to_numpy().T, rcond=None)
                except np.linalg.LinAlgError as e:
                    raise InversionError(f'{str(type(e))}: {str(e)}') from e
                R = R.T
            else:
                try:
                    R = solve(self.matrix, V.to_numpy().T).T
                except np.linalg.LinAlgError as e:
                    raise InversionError(f'{str(type(e))}: {str(e)}') from e

            if check_finite:
                if not np.isfinite(np.sum(R)):
                    raise InversionError('Result contains non-finite values')

            return self.source.make_array(R)

    def apply_inverse_adjoint(self, U, mu=None, initial_guess=None, least_squares=False):
        return self.H.apply_inverse(U, mu=mu, initial_guess=initial_guess, least_squares=least_squares)

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


class NumpyHankelOperator(NumpyGenericOperator):
    r"""Implicit representation of a Hankel operator by a |NumPy Array| of Markov parameters.

    Let

    .. math::
        h =
        \begin{pmatrix}
            h_1 & h_2 & \dots & h_n
        \end{pmatrix},\quad h_i\in\mathbb{C}^{p\times m},\,i=1,\,\dots,\,n,\quad n,m,p\in\mathbb{N}

    be a finite sequence of (matrix-valued) Markov parameters. For an odd number :math:`n=2s-1`
    of Markov parameters, the corresponding Hankel operator can be represented by the matrix

    .. math::
        H =
        \begin{bmatrix}
            h_1 & h_2 & \dots & h_s \\
            h_2 & h_3 & \dots & h_{s+1}\\
            \vdots & \vdots && \vdots\\
            h_s & h_{s+1} & \dots & h_{2s-1}
        \end{bmatrix}\in\mathbb{C}^{ms\times ps}.

    For an even number :math:`n=2s` of Markov parameters, the corresponding matrix
    representation is given by

    .. math::
        H =
        \begin{bmatrix}
            h_1 & h_2 & \dots & h_s & h_{s+1}\\
            h_2 & h_3 & \dots & h_{s+1} & h_{s+2}\\
            \vdots & \vdots && \vdots & \vdots\\
            h_s & h_{s+1} & \dots & h_{2s-1} & h_{2s}\\
            h_{s+1} & h_{s+2} & \dots & h_{2s} & 0
        \end{bmatrix}\in\mathbb{C}^{m(s+1)\times p(s+1)}.

    The matrix :math:`H` as seen above is not explicitly constructed, only the sequence of Markov
    parameters is stored. Efficient matrix-vector multiplications are realized via circulant
    matrices with DFT in the class' `apply` method
    (see :cite:`MSKC21` Algorithm 3.1. for details).

    Parameters
    ----------
    markov_parameters
        The |NumPy array| that contains the first :math:`n` Markov parameters that define the Hankel
        operator. Has to be one- or three-dimensional with either::

            markov_parameters.shape = (n,)

        for scalar-valued Markov parameters or::

            markov_parameters.shape = (n, p, m)

        for matrix-valued Markov parameters of dimension :math:`p\times m`.
    source_id
        The id of the operator's `source` |VectorSpace|.
    range_id
        The id of the operator's `range` |VectorSpace|.
    name
        Name of the operator.
    """

    def __init__(self, markov_parameters, source_id=None, range_id=None, name=None):
        if markov_parameters.ndim == 1:
            markov_parameters = markov_parameters.reshape(-1, 1, 1)
        assert markov_parameters.ndim == 3
        markov_parameters.setflags(write=False)  # make numpy arrays read-only
        self.__auto_init(locals())
        s, p, m = markov_parameters.shape
        n = s // 2 + 1
        self.source = NumpyVectorSpace(m * n, source_id)
        self.range = NumpyVectorSpace(p * n, range_id)
        self.linear = True
        self._circulant = self._calc_circulant()

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy().T
        k = U.shape[1]
        s, p, m = self.markov_parameters.shape
        n = s // 2 + 1

        FFT, iFFT = fft, ifft
        c = self._circulant
        dtype = complex
        if np.isrealobj(self.markov_parameters):
            if np.isrealobj(U):
                FFT, iFFT = rfft, irfft
                dtype = float
            else:
                c = np.concatenate([c, np.flip(c[1:-1], axis=0).conj()])

        y = np.zeros([self.range.dim, k], dtype=dtype)
        for (i, j) in np.ndindex((p, m)):
            x = np.concatenate([np.flip(U[j::m], axis=0), np.zeros([n, k])])
            cx = iFFT(FFT(x, axis=0) * c[:, i, j].reshape(-1, 1), axis=0)
            y[i::p] += cx[:n]

        return self.range.make_array(y.T)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    def _calc_circulant(self):
        FFT = rfft if np.isrealobj(self.markov_parameters) else fft
        s, p, m = self.markov_parameters.shape
        return FFT(
            np.roll(
                np.concatenate(
                    [
                        np.zeros([1, p, m]),
                        self.markov_parameters,
                        np.zeros([1 - s % 2, p, m]),
                    ]
                ),
                s // 2 + 1,
                axis=0,
            ),
            axis=0,
        )

    @property
    def H(self):
        adjoint_markov_parameters = self.markov_parameters.transpose(0, 2, 1).conj()
        return self.with_(
            markov_parameters=adjoint_markov_parameters,
            source_id=self.range_id,
            range_id=self.source_id,
            name=self.name + '_adjoint',
        )
