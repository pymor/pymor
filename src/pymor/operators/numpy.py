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
- |NumpyCirculantOperator| matrix-free operator of a circulant matrix.
- |NumpyToeplitzOperator| matrix-free operator of a Toeplitz matrix.
- |NumpyHankelOperator| matrix-free operator of a Hankel matrix.
"""

from functools import reduce

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps
from scipy.fft import fft, ifft, irfft, rfft
from scipy.io import mmwrite, savemat
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg.lapack import get_lapack_funcs

from pymor.core.base import abstractmethod
from pymor.core.cache import CacheableObject, cached
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
                 solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = NumpyVectorSpace(dim_source)
        self.range = NumpyVectorSpace(dim_range)
        self.parameters_own = parameters

    def apply(self, U, mu=None):
        assert U in self.source
        assert self.parameters.assert_compatible(mu)
        if self.parametric:
            return self.range.make_array(self.mapping(U.to_numpy_TP().T, mu=mu))
        else:
            return self.range.make_array(self.mapping(U.to_numpy_TP().T))

    def apply_adjoint(self, V, mu=None):
        if self.adjoint_mapping is None:
            raise ValueError('NumpyGenericOperator: adjoint mapping was not defined.')
        assert V in self.range
        assert self.parameters.assert_compatible(mu)
        V = V.to_numpy_TP().T
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

    .. note::
        In the case of a |NumPy array|, the `apply_inverse` method by default
        uses `check_finite=True` and `check_cond=True`.
        Setting them to `False` (e.g., via `defaults`) can significantly speed
        up the computation, especially for smaller matrices.

    Parameters
    ----------
    matrix
        The |NumPy array| or |SciPy spmatrix| which is to be wrapped.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, matrix, solver_options=None, name=None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        try:
            matrix.setflags(write=False)  # make numpy arrays read-only
        except AttributeError:
            pass

        self.__auto_init(locals())
        self.source = NumpyVectorSpace(matrix.shape[1])
        self.range = NumpyVectorSpace(matrix.shape[0])
        self.sparse = sps.issparse(matrix)

    @classmethod
    def from_file(cls, path, key=None, solver_options=None, name=None):
        from pymor.tools.io import load_matrix
        matrix = load_matrix(path, key=key)
        return cls(matrix, solver_options=solver_options, name=name or key or path)

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
        return self.with_(matrix=adjoint_matrix, solver_options=options, name=self.name + '_adjoint')

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
        return self.range.make_array(self.matrix.dot(U.to_numpy_TP()).T)

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @defaults('check_finite', 'check_cond', 'default_sparse_solver_backend')
    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False,
                      check_finite=True, check_cond=True, default_sparse_solver_backend='scipy'):
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
        check_cond
            Check condition number in case the matrix is a |NumPy array|.
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
                    R, _, _, _ = spla.lstsq(self.matrix, V.to_numpy_TP())
                except np.linalg.LinAlgError as e:
                    raise InversionError(f'{type(e)!s}: {e!s}') from e
                R = R.T
            else:
                if not hasattr(self, '_lu_factor'):
                    try:
                        self._lu_factor = lu_factor(self.matrix, check_finite=check_finite)
                    except np.linalg.LinAlgError as e:
                        raise InversionError(f'{type(e)!s}: {e!s}') from e
                    if check_cond:
                        gecon = get_lapack_funcs('gecon', self._lu_factor)
                        rcond, _ = gecon(self._lu_factor[0], np.linalg.norm(self.matrix, ord=1), norm='1')
                        if rcond < np.finfo(np.float64).eps:
                            self.logger.warning(f'Ill-conditioned matrix (rcond={rcond:.6g}) in apply_inverse: '
                                                'result may not be accurate.')
                R = lu_solve(self._lu_factor, V.to_numpy_TP(), check_finite=check_finite).T

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
                    matrix += (sps.eye(matrix.shape[0]) * identity_shift)
                except NotImplementedError:
                    matrix = matrix + (sps.eye(matrix.shape[0]) * identity_shift)
            else:
                matrix += (np.eye(matrix.shape[0]) * identity_shift)

        return NumpyMatrixOperator(matrix, solver_options=solver_options)

    def __getstate__(self):
        if hasattr(self.matrix, 'factorization'):  # remove unpicklable SuperLU factorization
            del self.matrix.factorization
        return self.__dict__

    def _format_repr(self, max_width, verbosity):
        if self.sparse:
            matrix_repr = f'<{self.range.dim}x{self.source.dim} sparse, {self.matrix.nnz} nnz>'
        else:
            matrix_repr = f'<{self.range.dim}x{self.source.dim} dense>'
        return super()._format_repr(max_width, verbosity, override={'matrix': matrix_repr})


class NumpyCirculantOperator(Operator, CacheableObject):
    r"""Matrix-free representation of a (block) circulant matrix by a |NumPy Array|.

    A (block) circulant matrix is a special kind of (block) Toeplitz matrix which is (block) square
    and completely determined by its first (matrix-valued) column via

    .. math::
        C =
            \begin{bmatrix}
                c_1    & c_n    & c_{n-1} & \cdots & \cdots & c_2 \\
                c_2    & c_1    & c_n     &        &        & \vdots \\
                c_3    & c_2    & \ddots  &        &        & \vdots \\
                \vdots &        &         & \ddots & c_n    & c_{n-1} \\
                \vdots &        &         & c_2    & c_1    & c_n \\
                c_n    & \cdots & \cdots  & c_3    & c_2    & c_1
            \end{bmatrix} \in \mathbb{C}^{n*p \times n*m},

    where the so-called circulant vector :math:`c \in \mathbb{C}^{\times n\times p\times m}` denotes
    the first (matrix-valued) column of the matrix. The matrix :math:`C` as seen above is not
    explicitly constructed, only `c` is stored. Efficient matrix-vector multiplications are realized
    with DFT in the class' `apply` method. See :cite:`GVL13` Chapter 4.8.2. for details.

    Parameters
    ----------
    c
        The |NumPy array| of shape `(n)` or `(n, p, m)` that defines the circulant vector.
    name
        Name of the operator.
    """

    cache_region = 'memory'
    linear = True

    def __init__(self, c, name=None):
        assert isinstance(c, np.ndarray)
        if c.ndim == 1:
            c = c.reshape(-1, 1, 1)
        assert c.ndim == 3
        c.setflags(write=False)  # make numpy arrays read-only
        self.__auto_init(locals())
        n, p, m = c.shape
        self._arr = c
        self.linear = True
        self.source = NumpyVectorSpace(n*m)
        self.range = NumpyVectorSpace(n*p)

    @cached
    def _circulant(self):
        return rfft(self._arr, axis=0) if np.isrealobj(self._arr) else fft(self._arr, axis=0)

    def _circular_matvec(self, vec):
        n, p, m = self._arr.shape
        s, k = vec.shape

        # use real arithmetic if possible
        isreal = np.isrealobj(self._arr) and np.isrealobj(vec)
        ismixed = np.isrealobj(self._arr) and np.iscomplexobj(vec)

        C = self._circulant()
        if ismixed:
            l =  s // m - C.shape[0] + 1
            C = np.concatenate([C, C[1:l].conj()[::-1]])

        dtype = float if isreal else complex
        y = np.zeros((self.range.dim, k), dtype=dtype)
        for j in range(m):
            x = vec[j::m]
            X = rfft(x, axis=0) if isreal else fft(x, axis=0)
            for i in range(p):
                Y = X*C[:, i, j].reshape(-1, 1)
                # setting n=n below is necessary to allow uneven lengths but considerably slower
                # Hankel operator will always pad to even length to avoid that
                Y = irfft(Y, n=n, axis=0) if isreal else ifft(Y, axis=0)
                y[i::p] += Y[:self.range.dim // p]
        return y.T

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy_TP()
        return self.range.make_array(self._circular_matvec(U))

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @property
    def H(self):
        return self.with_(c=np.roll(self._arr.conj(), -1, axis=0)[::-1].transpose(0, 2, 1), name=self.name + '_adjoint')


class NumpyToeplitzOperator(Operator):
    r"""Matrix-free representation of a finite dimensional Toeplitz matrix by a |NumPy Array|.

    A finite dimensional Toeplitz operator can be represented by a (block) matrix with constant
    diagonals (diagonal blocks), i.e.:

    .. math::
        T =
            \begin{bmatrix}
                c_1    & r_2    & r_3    & \cdots & \cdots & r_n \\
                c_2    & c_1    & r_2    &        &        & \vdots \\
                c_3    & c_2    & \ddots &        &        & \vdots \\
                \vdots &        &        & \ddots & r_2    & r_3 \\
                \vdots &        &        & c_2    & c_1    & r_2 \\
                c_m    & \cdots & \cdots & c_3    & c_2    & c_1
            \end{bmatrix} \in \mathbb{C}^{n*p \times k*m},

    where :math:`c\in\mathbb{C}^{n\times p\times m}` and :math:`r\in\mathbb{C}^{k\times p\times m}`
    denote the first (matrix-valued) column and first (matrix-valued) row of the (block) Toeplitz
    matrix, respectively. The matrix :math:`T` as seen above is not explicitly
    constructed, only the arrays `c` and `r` are stored. The operator's `apply` method takes
    advantage of the fact that any (block) Toeplitz matrix can be embedded in a larger (block)
    circulant matrix to leverage efficient matrix-vector multiplications with DFT.

    Parameters
    ----------
    c
        The |NumPy array| of shape either `(n)` or `(n, p, m)` that defines the first column of the
        (block) Toeplitz matrix.
    r
        The |NumPy array|  of shape `(k,)` or `(k, p, m)` that defines the first row of the Toeplitz
        matrix. If supplied, its first entry `r[0]` has to be equal to `c[0]`.
        Defaults to `None`. If `r` is `None`, the behavior of :func:`scipy.linalg.toeplitz` is
        mimicked which sets `r = c.conj()` (except for the first entry).
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, c, r=None, name=None):
        assert isinstance(c, np.ndarray)
        c = c.reshape(-1, 1, 1) if c.ndim == 1 else c
        assert c.ndim == 3
        if r is None:
            r = np.conjugate(c)
            r[0] = c[0]
        else:
            assert isinstance(r, np.ndarray)
            r = r.reshape(-1, 1, 1) if r.ndim == 1 else r
            assert r.ndim == 3
            assert c.shape[1:] == r.shape[1:]
            assert np.allclose(c[0], r[0])
        c.setflags(write=False)
        r.setflags(write=False)
        self.__auto_init(locals())
        self._circulant = NumpyCirculantOperator(
            np.concatenate([c, r[:0:-1]]),
            name=self.name + ' (implicit circulant)')
        _, p, m = self._circulant._arr.shape
        self.source = NumpyVectorSpace(m*r.shape[0])
        self.range = NumpyVectorSpace(p*c.shape[0])

    def apply(self, U, mu=None):
        assert U in self.source
        n, _, m = self._circulant._arr.shape
        U = np.concatenate([U.to_numpy_TP(), np.zeros((n*m - U.dim, len(U)))])
        return self.range.make_array(self._circulant._circular_matvec(U)[:, :self.range.dim])

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @property
    def H(self):
        return self.with_(c=self.r.conj().transpose(0, 2, 1), r=self.c.conj().transpose(0, 2, 1),
                          name=self.name + '_adjoint')


class NumpyHankelOperator(Operator):
    r"""Matrix-free representation of a finite dimensional Hankel operator by a |NumPy Array|.

    A finite dimensional Hankel operator can be represented by a (block) matrix with constant
    anti-diagonals (anti-diagonal blocks), i.e.:

    .. math::
        H =
            \begin{bmatrix}
                c_1    & c_2    & c_3    & \cdots  & \cdots  & r_1 \\
                c_2    & c_3    &        &         &         & \vdots \\
                c_3    &        &        &         &         & \vdots \\
                \vdots &        &        &         &         & r_{n-2} \\
                \vdots &        &        &         & r_{n-2} & r_{n-1} \\
                c_m    & \cdots & \cdots & r_{n-2} & r_{n-1} & r_n
            \end{bmatrix} \in \mathbb{C}^{n*p \times k*m},

    where :math:`c\in\mathbb{C}^{s\times p\times m}` and :math:`r\in\mathbb{C}^{k\times p\times m}`
    denote the first (matrix-valued) column and last (matrix-valued) row of the (block) Hankel
    matrix, respectively. The matrix :math:`H` as seen above is not explicitly constructed, only the
    arrays `c` and `r` are stored. Efficient matrix-vector multiplications are realized with DFT in
    the class' `apply` method (see :cite:`MSKC21` Algorithm 3.1. for details).

    Parameters
    ----------
    c
        The |NumPy array| of shape `(n)` or `(n, p, m)` that defines the first column of the
        (block) Hankel matrix.
    r
        The |NumPy array| of shape `(k,)` or `(k, p, m)` that defines the last row of the (block)
        Hankel matrix. If supplied, its first entry `r[0]` has to be equal to `c[-1]`.
        Defaults to `None`. If `r` is `None`, the behavior of :func:`scipy.linalg.hankel` is
        mimicked which sets `r` to zero (except for the first entry).
    name
        Name of the operator.
    """

    linear = True

    def __init__(self, c, r=None, name=None):
        assert isinstance(c, np.ndarray)
        c = c.reshape(-1, 1, 1) if c.ndim == 1 else c
        assert c.ndim == 3
        if r is None:
            r = np.zeros_like(c)
            r[0] = c[-1]
        else:
            assert isinstance(r, np.ndarray)
            r = r.reshape(-1, 1, 1) if r.ndim == 1 else r
            assert r.ndim == 3
            assert c.shape[1:] == r.shape[1:]
            assert np.allclose(r[0], c[-1])
        c.setflags(write=False)
        r.setflags(write=False)
        self.__auto_init(locals())
        k, l = c.shape[0], r.shape[0]
        n = k + l - 1
        # zero pad to even length if real to avoid slow irfft
        z = int(np.isrealobj(c) and np.isrealobj(r) and n % 2)
        h = np.concatenate((c, r[1:], np.zeros([z, *c.shape[1:]])))
        shift = n // 2 + int(np.ceil((k - l) / 2)) + (n % 2) + z # this works
        self._circulant = NumpyCirculantOperator(
            np.roll(h, shift, axis=0), name=self.name + ' (implicit circulant)')
        p, m = self._circulant._arr.shape[1:]
        self.source = NumpyVectorSpace(l*m)
        self.range = NumpyVectorSpace(k*p)

    def apply(self, U, mu=None):
        assert U in self.source
        U = U.to_numpy_TP()
        n, p, m = self._circulant._arr.shape
        x = np.zeros((n*m, U.shape[1]), dtype=U.dtype)
        for j in range(m):
            x[:self.source.dim][j::m] = np.flip(U[j::m], axis=0)
        return self.range.make_array(self._circulant._circular_matvec(x)[:, :self.range.dim])

    def apply_adjoint(self, V, mu=None):
        assert V in self.range
        return self.H.apply(V, mu=mu)

    @property
    def H(self):
        h = np.concatenate([self.c, self.r[1:]], axis=0).conj().transpose(0, 2, 1)
        return self.with_(c=h[:self.r.shape[0]], r=h[self.r.shape[0]-1:], name=self.name+'_adjoint')
