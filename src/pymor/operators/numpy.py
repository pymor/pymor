# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

"""This module provides the following |NumPy| based |Operators|:

  - |NumpyMatrixOperator| wraps a 2D |NumPy array| as a |Operator|.
  - |NumpyMatrixBasedOperator| should be used as base class for all |Operators|
    which assemble into a |NumpyMatrixOperator|.
  - |NumpyGenericOperator| wraps an arbitrary Python function between
    |NumPy arrays| as an |Operator|.
"""

from collections import OrderedDict
from functools import reduce

import numpy as np
import scipy.sparse
from scipy.sparse import issparse
from scipy.io import mmwrite, savemat

from pymor.algorithms import genericsolvers
from pymor.core.defaults import defaults, defaults_sid
from pymor.core.exceptions import InversionError
from pymor.core.interfaces import abstractmethod
from pymor.core.logger import getLogger
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import IdentityOperator, ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorArray, NumpyVectorSpace


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
    dim_source
        Dimension of the operator's source.
    dim_range
        Dimension of the operator's range.
    linear
        Set to `True` if the provided `mapping` is linear.
    parameter_type
        The |ParameterType| of the |Parameters| the mapping accepts.
    name
        Name of the operator.
    """

    def __init__(self, mapping, dim_source=1, dim_range=1, linear=False, parameter_type=None, solver_options=None,
                 name=None):
        self.source = NumpyVectorSpace(dim_source)
        self.range = NumpyVectorSpace(dim_range)
        self.solver_options = solver_options
        self.name = name
        self._mapping = mapping
        self.linear = linear
        if parameter_type is not None:
            self.build_parameter_type(parameter_type, local_global=True)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        if self.parametric:
            mu = self.parse_parameter(mu)
            return NumpyVectorArray(self._mapping(U_array, mu=mu), copy=False)
        else:
            return NumpyVectorArray(self._mapping(U_array), copy=False)


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
        The assembled **parameter independent** |Operator|.
        """
        if hasattr(self, '_assembled_operator'):
            if self._defaults_sid != defaults_sid():
                self.logger.warn('Re-assembling since state of global defaults has changed.')
                op = self._assembled_operator = NumpyMatrixOperator(self._assemble(),
                                                                    solver_options=self.solver_options)
                self._defaults_sid = defaults_sid()
                return op
            else:
                return self._assembled_operator
        elif not self.parameter_type:
            op = self._assembled_operator = NumpyMatrixOperator(self._assemble(), solver_options=self.solver_options)
            self._defaults_sid = defaults_sid()
            return op
        else:
            return NumpyMatrixOperator(self._assemble(self.parse_parameter(mu)), solver_options=self.solver_options)

    def apply(self, U, ind=None, mu=None):
        return self.assemble(mu).apply(U, ind=ind)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        return self.assemble(mu).apply_adjoint(U, ind=ind, source_product=source_product, range_product=range_product)

    def as_vector(self, mu=None):
        return self.assemble(mu).as_vector()

    def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, ind=ind, least_squares=least_squares)


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

    def __init__(self, matrix, solver_options=None, name=None):
        assert matrix.ndim <= 2
        if matrix.ndim == 1:
            matrix = np.reshape(matrix, (1, -1))
        self.source = NumpyVectorSpace(matrix.shape[1])
        self.range = NumpyVectorSpace(matrix.shape[0])
        self.solver_options = solver_options
        self.name = name
        self._matrix = matrix
        self.sparse = issparse(matrix)

    @classmethod
    def from_file(cls, path, key=None, solver_options=None, name=None):
        from pymor.tools.io import load_matrix
        matrix = load_matrix(path, key=key)
        return cls(matrix, solver_options=solver_options, name=name or key or path)

    def _assemble(self, mu=None):
        pass

    def assemble(self, mu=None):
        return self

    def as_vector(self, mu=None):
        matrix = self._matrix
        if matrix.shape[0] != 1 and matrix.shape[1] != 1:
            raise TypeError('This operator does not represent a vector or linear functional.')
        return NumpyVectorArray(matrix.ravel(), copy=True)

    def apply(self, U, ind=None, mu=None):
        assert U in self.source
        assert U.check_ind(ind)
        U_array = U._array[:U._len] if ind is None else U._array[ind]
        return NumpyVectorArray(self._matrix.dot(U_array.T).T, copy=False)

    def apply_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None):
        assert U in self.range
        assert U.check_ind(ind)
        assert source_product is None or source_product.source == source_product.range == self.source
        assert range_product is None or range_product.source == range_product.range == self.range
        if range_product:
            PrU = range_product.apply(U, ind=ind).data
        else:
            PrU = U.data if ind is None else U.data[ind]
        ATPrU = NumpyVectorArray(self._matrix.T.dot(PrU.T).T, copy=False)
        if source_product:
            return source_product.apply_inverse(ATPrU)
        else:
            return ATPrU

    def apply_inverse(self, V, ind=None, mu=None, least_squares=False):
        assert V in self.range
        assert V.check_ind(ind)

        if V.dim == 0:
            if self.source.dim == 0 or least_squares:
                return NumpyVectorArray(np.zeros((V.len_ind(ind), self.source.dim)))
            else:
                raise InversionError

        options = (self.solver_options.get('inverse') if self.solver_options else
                   'least_squares' if least_squares else
                   None)

        if options and not least_squares:
            solver_type = options if isinstance(options, str) else options['type']
            if solver_type.startswith('least_squares'):
                self.logger.warn('Least squares solver selected but "least_squares == False"')

        V = V.data if ind is None else \
            V.data[ind] if hasattr(ind, '__len__') else V.data[ind:ind + 1]

        try:
            return NumpyVectorArray(_apply_inverse(self._matrix, V, options=options), copy=False)
        except InversionError as e:
            if least_squares and options:
                solver_type = options if isinstance(options, str) else options['type']
                if not solver_type.startswith('least_squares'):
                    msg = str(e) \
                        + '\nNote: linear solver was selected for solving least squares problem (maybe not invertible?)'
                    raise InversionError(msg)
            raise e

    def apply_inverse_adjoint(self, U, ind=None, mu=None, source_product=None, range_product=None,
                              least_squares=False):
        if source_product or range_product:
            return super().apply_inverse_adjoint(U, ind=ind, mu=mu,
                                                 source_product=source_product,
                                                 range_product=range_product,
                                                 least_squares=least_squares)
        else:
            options = {'inverse': self.solver_options.get('inverse_adjoint') if self.solver_options else None}
            adjoint_op = NumpyMatrixOperator(self._matrix.T, solver_options=options)
            return adjoint_op.apply_inverse(U, ind=ind, mu=mu, least_squares=least_squares)

    def projected_to_subbasis(self, dim_range=None, dim_source=None, name=None):
        """Project the operator to a subbasis.

        The purpose of this method is to further project an operator that has been
        obtained through :meth:`~pymor.operators.interfaces.OperatorInterface.projected`
        to subbases of the original projection bases, i.e. ::

            op.projected(r_basis, s_basis, prod).projected_to_subbasis(dim_range, dim_source)

        should be the same as ::

            op.projected(r_basis.copy(range(dim_range)), s_basis.copy(range(dim_source)), prod)

        For a |NumpyMatrixOperator| this amounts to extracting the upper-left
        (dim_range, dim_source) corner of the matrix it wraps.

        Parameters
        ----------
        dim_range
            Dimension of the range subbasis.
        dim_source
            Dimension of the source subbasis.

        Returns
        -------
        The projected |Operator|.
        """
        assert dim_source is None or dim_source <= self.source.dim
        assert dim_range is None or dim_range <= self.range.dim
        name = name or '{}_projected_to_subbasis'.format(self.name)
        # copy instead of just slicing the matrix to ensure contiguous memory
        return NumpyMatrixOperator(self._matrix[:dim_range, :dim_source].copy(), solver_options=self.solver_options,
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
        return NumpyMatrixOperator(matrix, solver_options=solver_options)

    def __getstate__(self):
        if hasattr(self._matrix, 'factorization'):  # remove unplicklable SuperLU factorization
            del self._matrix.factorization
        return self.__dict__


####################################################################################################


import scipy.version
from scipy.sparse.linalg import bicgstab, spsolve, splu, spilu, lgmres, lsqr, LinearOperator
try:
    from scipy.sparse.linalg import lsmr
    HAVE_SCIPY_LSMR = True
except ImportError:
    HAVE_SCIPY_LSMR = False

try:
    import pyamg
    HAVE_PYAMG = True
except ImportError:
    HAVE_PYAMG = False


_dense_options = None
_dense_options_sid = None
_sparse_options = None
_sparse_options_sid = None


@defaults('default_solver', 'default_least_squares_solver', 'least_squares_lstsq_rcond')
def dense_options(default_solver='solve',
                  default_least_squares_solver='least_squares_lstsq',
                  least_squares_lstsq_rcond=-1.):
    """Returns |solver_options| (with default values) for dense |NumPy| matricies.

    Parameters
    ----------
    default_solver
        Default dense solver to use (solve, least_squares_lstsq, generic_lgmres,
        least_squares_generic_lsmr, least_squares_generic_lsqr).
    default_least_squares_solver
        Default solver to use for least squares problems (least_squares_lstsq,
        least_squares_generic_lsmr, least_squares_generic_lsqr).
    least_squares_lstsq_rcond
        See :func:`numpy.linalg.lstsq`.

    Returns
    -------
    A tuple of possible values for |solver_options|.
    """

    assert default_least_squares_solver.startswith('least_squares')

    opts = (('solve',               {'type': 'solve'}),
            ('least_squares_lstsq', {'type': 'least_squares_lstsq',
                                     'rcond': least_squares_lstsq_rcond}))
    opts = OrderedDict(opts)
    opts.update(genericsolvers.options())
    def_opt = opts.pop(default_solver)
    if default_least_squares_solver != default_solver:
        def_ls_opt = opts.pop(default_least_squares_solver)
        ordered_opts = OrderedDict(((default_solver, def_opt),
                                    (default_least_squares_solver, def_ls_opt)))
    else:
        ordered_opts = OrderedDict(((default_solver, def_opt),))
    ordered_opts.update(opts)
    return ordered_opts


@defaults('default_solver', 'default_least_squares_solver', 'bicgstab_tol', 'bicgstab_maxiter', 'spilu_drop_tol',
          'spilu_fill_factor', 'spilu_drop_rule', 'spilu_permc_spec', 'spsolve_permc_spec',
          'spsolve_keep_factorization',
          'lgmres_tol', 'lgmres_maxiter', 'lgmres_inner_m', 'lgmres_outer_k', 'least_squares_lsmr_damp',
          'least_squares_lsmr_atol', 'least_squares_lsmr_btol', 'least_squares_lsmr_conlim',
          'least_squares_lsmr_maxiter', 'least_squares_lsmr_show', 'least_squares_lsqr_atol',
          'least_squares_lsqr_btol', 'least_squares_lsqr_conlim', 'least_squares_lsqr_iter_lim',
          'least_squares_lsqr_show', 'pyamg_tol', 'pyamg_maxiter', 'pyamg_verb', 'pyamg_rs_strength', 'pyamg_rs_CF',
          'pyamg_rs_postsmoother', 'pyamg_rs_max_levels', 'pyamg_rs_max_coarse', 'pyamg_rs_coarse_solver',
          'pyamg_rs_cycle', 'pyamg_rs_accel', 'pyamg_rs_tol', 'pyamg_rs_maxiter',
          'pyamg_sa_symmetry', 'pyamg_sa_strength', 'pyamg_sa_aggregate', 'pyamg_sa_smooth',
          'pyamg_sa_presmoother', 'pyamg_sa_postsmoother', 'pyamg_sa_improve_candidates', 'pyamg_sa_max_levels',
          'pyamg_sa_max_coarse', 'pyamg_sa_diagonal_dominance', 'pyamg_sa_coarse_solver', 'pyamg_sa_cycle',
          'pyamg_sa_accel', 'pyamg_sa_tol', 'pyamg_sa_maxiter',
          sid_ignore=('least_squares_lsmr_show', 'least_squares_lsqr_show', 'pyamg_verb'))
def sparse_options(default_solver='spsolve',
                   default_least_squares_solver='least_squares_lsmr' if HAVE_SCIPY_LSMR else 'least_squares_generic_lsmr',
                   bicgstab_tol=1e-15,
                   bicgstab_maxiter=None,
                   spilu_drop_tol=1e-4,
                   spilu_fill_factor=10,
                   spilu_drop_rule='basic,area',
                   spilu_permc_spec='COLAMD',
                   spsolve_permc_spec='COLAMD',
                   spsolve_keep_factorization=True,
                   lgmres_tol=1e-5,
                   lgmres_maxiter=1000,
                   lgmres_inner_m=39,
                   lgmres_outer_k=3,
                   least_squares_lsmr_damp=0.0,
                   least_squares_lsmr_atol=1e-6,
                   least_squares_lsmr_btol=1e-6,
                   least_squares_lsmr_conlim=1e8,
                   least_squares_lsmr_maxiter=None,
                   least_squares_lsmr_show=False,
                   least_squares_lsqr_damp=0.0,
                   least_squares_lsqr_atol=1e-6,
                   least_squares_lsqr_btol=1e-6,
                   least_squares_lsqr_conlim=1e8,
                   least_squares_lsqr_iter_lim=None,
                   least_squares_lsqr_show=False,
                   pyamg_tol=1e-5,
                   pyamg_maxiter=400,
                   pyamg_verb=False,
                   pyamg_rs_strength=('classical', {'theta': 0.25}),
                   pyamg_rs_CF='RS',
                   pyamg_rs_presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_rs_postsmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_rs_max_levels=10,
                   pyamg_rs_max_coarse=500,
                   pyamg_rs_coarse_solver='pinv2',
                   pyamg_rs_cycle='V',
                   pyamg_rs_accel=None,
                   pyamg_rs_tol=1e-5,
                   pyamg_rs_maxiter=100,
                   pyamg_sa_symmetry='hermitian',
                   pyamg_sa_strength='symmetric',
                   pyamg_sa_aggregate='standard',
                   pyamg_sa_smooth=('jacobi', {'omega': 4.0/3.0}),
                   pyamg_sa_presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_sa_postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
                   pyamg_sa_improve_candidates=[('block_gauss_seidel', {'sweep': 'symmetric', 'iterations': 4}), None],
                   pyamg_sa_max_levels=10,
                   pyamg_sa_max_coarse=500,
                   pyamg_sa_diagonal_dominance=False,
                   pyamg_sa_coarse_solver='pinv2',
                   pyamg_sa_cycle='V',
                   pyamg_sa_accel=None,
                   pyamg_sa_tol=1e-5,
                   pyamg_sa_maxiter=100):
    """Returns |solver_options| (with default values) for sparse |NumPy| matricies.

    Parameters
    ----------
    default_solver
        Default sparse solver to use (spsolve, bicgstab, bicgstab_spilu, pyamg,
        pyamg_rs, pyamg_sa, generic_lgmres, least_squares_lsmr, least_squares_lsqr).
    default_least_squares_solver
        Default solver to use for least squares problems (least_squares_lsmr,
        least_squares_lsqr).
    bicgstab_tol
        See :func:`scipy.sparse.linalg.bicgstab`.
    bicgstab_maxiter
        See :func:`scipy.sparse.linalg.bicgstab`.
    spilu_drop_tol
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_fill_factor
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_drop_rule
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_permc_spec
        See :func:`scipy.sparse.linalg.spilu`.
    spsolve_permc_spec
        See :func:`scipy.sparse.linalg.spsolve`.
    lgmres_tol
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_maxiter
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_inner_m
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_outer_k
        See :func:`scipy.sparse.linalg.lgmres`.
    least_squares_lsmr_damp
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_atol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_btol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_conlim
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_maxiter
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_show
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsqr_damp
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_atol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_btol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_conlim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_iter_lim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_show
        See :func:`scipy.sparse.linalg.lsqr`.
    pyamg_tol
        Tolerance for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
    pyamg_maxiter
        Maximum iterations for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
    pyamg_verb
        Verbosity flag for `PyAMG <http://pyamg.github.io/>`_ blackbox solver.
    pyamg_rs_strength
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_CF
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_presmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_postsmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_max_levels
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_max_coarse
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_coarse_solver
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_cycle
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_accel
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_tol
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_rs_maxiter
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Ruge-Stuben solver.
    pyamg_sa_symmetry
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_strength
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_aggregate
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_smooth
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_presmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_postsmoother
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_improve_candidates
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_max_levels
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_max_coarse
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_diagonal_dominance
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_coarse_solver
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_cycle
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_accel
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_tol
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.
    pyamg_sa_maxiter
        Parameter for `PyAMG <http://pyamg.github.io/>`_ Smoothed-Aggregation solver.

    Returns
    -------
    A tuple of all possible |solver_options|.
    """

    assert default_least_squares_solver.startswith('least_squares')

    opts = (('bicgstab_spilu',     {'type': 'bicgstab_spilu',
                                    'tol': bicgstab_tol,
                                    'maxiter': bicgstab_maxiter,
                                    'spilu_drop_tol': spilu_drop_tol,
                                    'spilu_fill_factor': spilu_fill_factor,
                                    'spilu_drop_rule': spilu_drop_rule,
                                    'spilu_permc_spec': spilu_permc_spec}),
            ('bicgstab',           {'type': 'bicgstab',
                                    'tol': bicgstab_tol,
                                    'maxiter': bicgstab_maxiter}),
            ('spsolve',            {'type': 'spsolve',
                                    'permc_spec': spsolve_permc_spec,
                                    'keep_factorization': spsolve_keep_factorization}),
            ('lgmres',             {'type': 'lgmres',
                                    'tol': lgmres_tol,
                                    'maxiter': lgmres_maxiter,
                                    'inner_m': lgmres_inner_m,
                                    'outer_k': lgmres_outer_k}),
            ('least_squares_lsqr', {'type': 'least_squares_lsqr',
                                    'damp': least_squares_lsqr_damp,
                                    'atol': least_squares_lsqr_atol,
                                    'btol': least_squares_lsqr_btol,
                                    'conlim': least_squares_lsqr_conlim,
                                    'iter_lim': least_squares_lsqr_iter_lim,
                                    'show': least_squares_lsqr_show}))

    if HAVE_SCIPY_LSMR:
        opts += (('least_squares_lsmr', {'type': 'least_squares_lsmr',
                                         'damp': least_squares_lsmr_damp,
                                         'atol': least_squares_lsmr_atol,
                                         'btol': least_squares_lsmr_btol,
                                         'conlim': least_squares_lsmr_conlim,
                                         'maxiter': least_squares_lsmr_maxiter,
                                         'show': least_squares_lsmr_show}),)

    if HAVE_PYAMG:
        opts += (('pyamg',    {'type': 'pyamg',
                               'tol': pyamg_tol,
                               'maxiter': pyamg_maxiter}),
                 ('pyamg-rs', {'type': 'pyamg-rs',
                               'strength': pyamg_rs_strength,
                               'CF': pyamg_rs_CF,
                               'presmoother': pyamg_rs_presmoother,
                               'postsmoother': pyamg_rs_postsmoother,
                               'max_levels': pyamg_rs_max_levels,
                               'max_coarse': pyamg_rs_max_coarse,
                               'coarse_solver': pyamg_rs_coarse_solver,
                               'cycle': pyamg_rs_cycle,
                               'accel': pyamg_rs_accel,
                               'tol': pyamg_rs_tol,
                               'maxiter': pyamg_rs_maxiter}),
                 ('pyamg-sa', {'type': 'pyamg-sa',
                               'symmetry': pyamg_sa_symmetry,
                               'strength': pyamg_sa_strength,
                               'aggregate': pyamg_sa_aggregate,
                               'smooth': pyamg_sa_smooth,
                               'presmoother': pyamg_sa_presmoother,
                               'postsmoother': pyamg_sa_postsmoother,
                               'improve_candidates': pyamg_sa_improve_candidates,
                               'max_levels': pyamg_sa_max_levels,
                               'max_coarse': pyamg_sa_max_coarse,
                               'diagonal_dominance': pyamg_sa_diagonal_dominance,
                               'coarse_solver': pyamg_sa_coarse_solver,
                               'cycle': pyamg_sa_cycle,
                               'accel': pyamg_sa_accel,
                               'tol': pyamg_sa_tol,
                               'maxiter': pyamg_sa_maxiter}))
    opts = OrderedDict(opts)
    opts.update(genericsolvers.options())
    def_opt = opts.pop(default_solver)
    if default_least_squares_solver != default_solver:
        def_ls_opt = opts.pop(default_least_squares_solver)
        ordered_opts = OrderedDict(((default_solver, def_opt),
                                    (default_least_squares_solver, def_ls_opt)))
    else:
        ordered_opts = OrderedDict(((default_solver, def_opt),))
    ordered_opts.update(opts)
    return ordered_opts


def _options(matrix=None, sparse=None):
    """Returns |solver_options| (with default values) for a given |NumPy| matrix.

    See :func:`dense_options` for documentation of all possible options for
    dense matrices.

    See :func:`sparse_options` for documentation of all possible options for
    sparse matrices.

    Parameters
    ----------
    matrix
        The matrix for which to return the options.
    sparse
        Instead of providing a matrix via the `matrix` argument,
        `sparse` can be set to `True` or `False` to requset the
        invert options for sparse or dense matrices.

    Returns
    -------
    A tuple of all possible |solver_options|.
    """
    global _dense_options, _dense_options_sid, _sparse_options, _sparse_options_sid
    assert (matrix is None) != (sparse is None)
    sparse = sparse if sparse is not None else issparse(matrix)
    if sparse:
        if not _sparse_options or _sparse_options_sid != defaults_sid():
            _sparse_options = sparse_options()
            _sparse_options_sid = defaults_sid()
            return _sparse_options
        else:
            return _sparse_options
    else:
        if not _dense_options or _dense_options_sid != defaults_sid():
            _dense_options = dense_options()
            _dense_options_sid = defaults_sid()
            return _dense_options
        else:
            return _dense_options


def _apply_inverse(matrix, V, options=None):
    """Solve linear equation system.

    Applies the inverse of `matrix` to the row vectors in `V`.

    See :func:`dense_options` for documentation of all possible options for
    sparse matrices.

    See :func:`sparse_options` for documentation of all possible options for
    sparse matrices.

    This method is called by :meth:`pymor.core.NumpyMatrixOperator.apply_inverse`
    and usually should not be used directly.

    Parameters
    ----------
    matrix
        The |NumPy| matrix to invert.
    V
        2-dimensional |NumPy array| containing as row vectors
        the right-hand sides of the linear equation systems to
        solve.
    options
        The solver options to use. (See :func:`_options`.)

    Returns
    -------
    |NumPy array| of the solution vectors.
    """

    default_options = _options(matrix)

    if options is None:
        options = next(iter(default_options.values()))
    elif isinstance(options, str):
        if options == 'least_squares':
            for k, v in default_options.items():
                if k.startswith('least_squares'):
                    options = v
                    break
            assert not isinstance(options, str)
        else:
            options = default_options[options]
    else:
        assert 'type' in options and options['type'] in default_options \
            and options.keys() <= default_options[options['type']].keys()
        user_options = options
        options = default_options[user_options['type']]
        options.update(user_options)

    promoted_type = np.promote_types(matrix.dtype, V.dtype)
    R = np.empty((len(V), matrix.shape[1]), dtype=promoted_type)

    if options['type'] == 'solve':
        for i, VV in enumerate(V):
            try:
                R[i] = np.linalg.solve(matrix, VV)
            except np.linalg.LinAlgError as e:
                raise InversionError('{}: {}'.format(str(type(e)), str(e)))
    elif options['type'] == 'least_squares_lstsq':
        for i, VV in enumerate(V):
            try:
                R[i], _, _, _ = np.linalg.lstsq(matrix, VV, rcond=options['rcond'])
            except np.linalg.LinAlgError as e:
                raise InversionError('{}: {}'.format(str(type(e)), str(e)))
    elif options['type'] == 'bicgstab':
        for i, VV in enumerate(V):
            R[i], info = bicgstab(matrix, VV, tol=options['tol'], maxiter=options['maxiter'])
            if info != 0:
                if info > 0:
                    raise InversionError('bicgstab failed to converge after {} iterations'.format(info))
                else:
                    raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                         format(info))
    elif options['type'] == 'bicgstab_spilu':
        # workaround for https://github.com/pymor/pymor/issues/171
        try:
            ilu = spilu(matrix, drop_tol=options['spilu_drop_tol'], fill_factor=options['spilu_fill_factor'],
                        drop_rule=options['spilu_drop_rule'], permc_spec=options['spilu_permc_spec'])
        except TypeError as t:
            logger = getLogger('pymor.operators.numpy._apply_inverse')
            logger.error("ignoring drop_rule in ilu factorization")
            ilu = spilu(matrix, drop_tol=options['spilu_drop_tol'], fill_factor=options['spilu_fill_factor'],
                        permc_spec=options['spilu_permc_spec'])
        precond = LinearOperator(matrix.shape, ilu.solve)
        for i, VV in enumerate(V):
            R[i], info = bicgstab(matrix, VV, tol=options['tol'], maxiter=options['maxiter'], M=precond)
            if info != 0:
                if info > 0:
                    raise InversionError('bicgstab failed to converge after {} iterations'.format(info))
                else:
                    raise InversionError('bicgstab failed with error code {} (illegal input or breakdown)'.
                                         format(info))
    elif options['type'] == 'spsolve':
        try:
            # maybe remove unusable factorization:
            if hasattr(matrix, 'factorization'):
                fdtype = matrix.factorizationdtype
                if not np.can_cast(V.dtype, fdtype, casting='safe'):
                    del matrix.factorization

            if list(map(int, scipy.version.version.split('.'))) >= [0, 14, 0]:
                if hasattr(matrix, 'factorization'):
                    # we may use a complex factorization of a real matrix to
                    # apply it to a real vector. In that case, we downcast
                    # the result here, removing the imaginary part,
                    # which should be zero.
                    R = matrix.factorization.solve(V.T).T.astype(promoted_type, copy=False)
                elif options['keep_factorization']:
                    # the matrix is always converted to the promoted type.
                    # if matrix.dtype == promoted_type, this is a no_op
                    matrix.factorization = splu(matrix_astype_nocopy(matrix, promoted_type), permc_spec=options['permc_spec'])
                    matrix.factorizationdtype = promoted_type
                    R = matrix.factorization.solve(V.T).T
                else:
                    # the matrix is always converted to the promoted type.
                    # if matrix.dtype == promoted_type, this is a no_op
                    R = spsolve(matrix_astype_nocopy(matrix, promoted_type), V.T, permc_spec=options['permc_spec']).T
            else:
                # see if-part for documentation
                if hasattr(matrix, 'factorization'):
                    for i, VV in enumerate(V):
                        R[i] = matrix.factorization.solve(VV).astype(promoted_type, copy=False)
                elif options['keep_factorization']:
                    matrix.factorization = splu(matrix_astype_nocopy(matrix, promoted_type), permc_spec=options['permc_spec'])
                    matrix.factorizationdtype = promoted_type
                    for i, VV in enumerate(V):
                        R[i] = matrix.factorization.solve(VV)
                elif len(V) > 1:
                    factorization = splu(matrix_astype_nocopy(matrix, promoted_type), permc_spec=options['permc_spec'])
                    for i, VV in enumerate(V):
                        R[i] = factorization.solve(VV)
                else:
                    R = spsolve(matrix_astype_nocopy(matrix, promoted_type), V.T, permc_spec=options['permc_spec']).reshape((1, -1))
        except RuntimeError as e:
            raise InversionError(e)
    elif options['type'] == 'lgmres':
        for i, VV in enumerate(V):
            R[i], info = lgmres(matrix, VV.copy(i),
                                tol=options['tol'],
                                maxiter=options['maxiter'],
                                inner_m=options['inner_m'],
                                outer_k=options['outer_k'])
            if info > 0:
                raise InversionError('lgmres failed to converge after {} iterations'.format(info))
            assert info == 0
    elif options['type'] == 'least_squares_lsmr':
        for i, VV in enumerate(V):
            R[i], info, itn, _, _, _, _, _ = lsmr(matrix, VV.copy(i),
                                                  damp=options['damp'],
                                                  atol=options['atol'],
                                                  btol=options['btol'],
                                                  conlim=options['conlim'],
                                                  maxiter=options['maxiter'],
                                                  show=options['show'])
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError('lsmr failed to converge after {} iterations'.format(itn))
    elif options['type'] == 'least_squares_lsqr':
        for i, VV in enumerate(V):
            R[i], info, itn, _, _, _, _, _, _, _ = lsqr(matrix, VV.copy(i),
                                                        damp=options['damp'],
                                                        atol=options['atol'],
                                                        btol=options['btol'],
                                                        conlim=options['conlim'],
                                                        iter_lim=options['iter_lim'],
                                                        show=options['show'])
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError('lsmr failed to converge after {} iterations'.format(itn))
    elif options['type'] == 'pyamg':
        if len(V) > 0:
            V_iter = iter(enumerate(V))
            R[0], ml = pyamg.solve(matrix, next(V_iter)[1],
                                   tol=options['tol'],
                                   maxiter=options['maxiter'],
                                   return_solver=True)
            for i, VV in V_iter:
                R[i] = pyamg.solve(matrix, VV,
                                   tol=options['tol'],
                                   maxiter=options['maxiter'],
                                   existing_solver=ml)
    elif options['type'] == 'pyamg-rs':
        ml = pyamg.ruge_stuben_solver(matrix,
                                      strength=options['strength'],
                                      CF=options['CF'],
                                      presmoother=options['presmoother'],
                                      postsmoother=options['postsmoother'],
                                      max_levels=options['max_levels'],
                                      max_coarse=options['max_coarse'],
                                      coarse_solver=options['coarse_solver'])
        for i, VV in enumerate(V):
            R[i] = ml.solve(VV,
                            tol=options['tol'],
                            maxiter=options['maxiter'],
                            cycle=options['cycle'],
                            accel=options['accel'])
    elif options['type'] == 'pyamg-sa':
        ml = pyamg.smoothed_aggregation_solver(matrix,
                                               symmetry=options['symmetry'],
                                               strength=options['strength'],
                                               aggregate=options['aggregate'],
                                               smooth=options['smooth'],
                                               presmoother=options['presmoother'],
                                               postsmoother=options['postsmoother'],
                                               improve_candidates=options['improve_candidates'],
                                               max_levels=options['max_levels'],
                                               max_coarse=options['max_coarse'],
                                               diagonal_dominance=options['diagonal_dominance'])
        for i, VV in enumerate(V):
            R[i] = ml.solve(VV,
                            tol=options['tol'],
                            maxiter=options['maxiter'],
                            cycle=options['cycle'],
                            accel=options['accel'])
    elif options['type'].startswith('generic') or options['type'].startswith('least_squares_generic'):
        logger = getLogger('pymor.operators.numpy._apply_inverse')
        logger.warn('You have selected a (potentially slow) generic solver for a NumPy matrix operator!')
        from pymor.operators.numpy import NumpyMatrixOperator
        from pymor.vectorarrays.numpy import NumpyVectorArray
        return genericsolvers.apply_inverse(NumpyMatrixOperator(matrix),
                                            NumpyVectorArray(V, copy=False),
                                            options=options).data
    else:
        raise ValueError('Unknown solver type')
    return R


# unfortunately, this is necessary, as scipy does not
# forward the copy=False argument in its csc_matrix.astype function
def matrix_astype_nocopy(matrix, dtype):
    if matrix.dtype == dtype:
        return matrix
    else:
        return matrix.astype(dtype)
