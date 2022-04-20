# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
config.require('PYMESS')


import numpy as np
import scipy.linalg as spla
import pymess

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.lyapunov import (mat_eqn_sparse_min_size, _solve_lyap_lrcf_check_args,
                                       _solve_lyap_dense_check_args, _chol)
from pymor.algorithms.to_matrix import to_matrix
from pymor.bindings.scipy import _solve_ricc_check_args
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import IdentityOperator


@defaults('adi_maxit', 'adi_memory_usage', 'adi_output', 'adi_rel_change_tol', 'adi_res2_tol', 'adi_res2c_tol',
          'adi_shifts_arp_m', 'adi_shifts_arp_p', 'adi_shifts_b0', 'adi_shifts_l0', 'adi_shifts_p',
          'adi_shifts_paratype')
def lradi_solver_options(adi_maxit=500,
                         adi_memory_usage=pymess.MESS_MEMORY_MID,
                         adi_output=1,
                         adi_rel_change_tol=1e-10,
                         adi_res2_tol=1e-10,
                         adi_res2c_tol=1e-11,
                         adi_shifts_arp_m=32,
                         adi_shifts_arp_p=48,
                         adi_shifts_b0=None,
                         adi_shifts_l0=16,
                         adi_shifts_p=None,
                         adi_shifts_paratype=pymess.MESS_LRCFADI_PARA_ADAPTIVE_Z):
    """Return available adi solver options with default values for the pymess backend.

    Parameters
    ----------
    adi_maxit
        See `pymess.OptionsAdi`.
    adi_memory_usage
        See `pymess.OptionsAdi`.
    adi_output
        See `pymess.OptionsAdi`.
    adi_rel_change_tol
        See `pymess.OptionsAdi`.
    adi_res2_tol
        See `pymess.OptionsAdi`.
    adi_res2c_tol
        See `pymess.OptionsAdi`.
    adi_shifts_arp_m
        See `pymess.OptionsAdiShifts`.
    adi_shifts_arp_p
        See `pymess.OptionsAdiShifts`.
    adi_shifts_b0
        See `pymess.OptionsAdiShifts`.
    adi_shifts_l0
        See `pymess.OptionsAdiShifts`.
    adi_shifts_p
        See `pymess.OptionsAdiShifts`.
    adi_shifts_paratype
        See `pymess.OptionsAdiShifts`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    lradi_opts = pymess.Options()
    lradi_opts.adi.maxit = adi_maxit
    lradi_opts.adi.memory_usage = adi_memory_usage
    lradi_opts.adi.output = adi_output
    lradi_opts.adi.rel_change_tol = adi_rel_change_tol
    lradi_opts.adi.res2_tol = adi_res2_tol
    lradi_opts.adi.res2c_tol = adi_res2c_tol
    lradi_opts.adi.shifts.arp_m = adi_shifts_arp_m
    lradi_opts.adi.shifts.arp_p = adi_shifts_arp_p
    lradi_opts.adi.shifts.b0 = adi_shifts_b0
    lradi_opts.adi.shifts.l0 = adi_shifts_l0
    lradi_opts.adi.shifts.p = adi_shifts_p
    lradi_opts.adi.shifts.paratype = adi_shifts_paratype
    return lradi_opts


def lyap_lrcf_solver_options():
    """Return available Lyapunov solvers with default options for the pymess backend.

    Also see :func:`lradi_solver_options`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'pymess_glyap': {'type': 'pymess_glyap'},
            'pymess_lradi': {'type': 'pymess_lradi',
                             'opts': lradi_solver_options()}}


@defaults('default_solver')
def solve_lyap_lrcf(A, E, B, trans=False, cont_time=True, options=None, default_solver=None):
    """Compute an approximate low-rank solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_lrcf`

    for a general description.

    This function uses `pymess.glyap` and `pymess.lradi`.
    For both methods,
    :meth:`~pymor.vectorarrays.interface.VectorArray.to_numpy`
    and
    :meth:`~pymor.vectorarrays.interface.VectorSpace.from_numpy`
    need to be implemented for `A.source`.
    Additionally, since `glyap` is a dense solver, it expects
    :func:`~pymor.algorithms.to_matrix.to_matrix` to work for A and E.

    If the solver is not specified using the options or default_solver arguments, `glyap` is used
    for small problems (smaller than defined with
    :func:`~pymor.algorithms.lyapunov.mat_eqn_sparse_min_size`) and `lradi` for large problems.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    trans
        Whether the first |Operator| in the Lyapunov equation is transposed.
    cont_time
        Whether the continuous- or discrete-time Lyapunov equation is solved.
        Only the continuous-time case is implemented.
    options
        The solver options to use (see :func:`lyap_lrcf_solver_options`).
    default_solver
        Default solver to use (pymess_lradi, pymess_glyap).
        If `None`, choose solver depending on the dimension of A.

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    _solve_lyap_lrcf_check_args(A, E, B, trans)
    if default_solver is None:
        default_solver = 'pymess_lradi' if A.source.dim >= mat_eqn_sparse_min_size() else 'pymess_glyap'
    options = _parse_options(options, lyap_lrcf_solver_options(), default_solver, None, False)

    if options['type'] == 'pymess_glyap':
        X = solve_lyap_dense(to_matrix(A, format='dense'),
                             to_matrix(E, format='dense') if E else None,
                             B.to_numpy().T if not trans else B.to_numpy(),
                             trans=trans, cont_time=cont_time, options=options)
        Z = _chol(X)
    elif options['type'] == 'pymess_lradi':
        opts = options['opts']
        opts.type = pymess.MESS_OP_NONE if not trans else pymess.MESS_OP_TRANSPOSE
        eqn = LyapunovEquation(opts, A, E, B)
        Z, status = pymess.lradi(eqn, opts)
        relres = status.res2_norm / status.res2_0
        if relres > opts.adi.res2_tol:
            logger = getLogger('pymor.bindings.pymess.solve_lyap_lrcf')
            logger.warning(f'Desired relative residual tolerance was not achieved '
                           f'({relres:e} > {opts.adi.res2_tol:e}).')
    else:
        raise ValueError(f'Unexpected Lyapunov equation solver ({options["type"]}).')

    return A.source.from_numpy(Z.T)


def lyap_dense_solver_options():
    """Return available Lyapunov solvers with default options for the pymess backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'pymess_glyap': {'type': 'pymess_glyap'}}


def solve_lyap_dense(A, E, B, trans=False, cont_time=True, options=None):
    """Compute the solution of a Lyapunov equation.

    See

    - :func:`pymor.algorithms.lyapunov.solve_cont_lyap_dense`

    for a general description.

    This function uses `pymess.glyap`.

    Parameters
    ----------
    A
        The matrix A as a 2D |NumPy array|.
    E
        The matrix E as a 2D |NumPy array| or `None`.
    B
        The matrix B as a 2D |NumPy array|.
    trans
        Whether the first operator in the Lyapunov equation is transposed.
    cont_time
        Whether the continuous- or discrete-time Lyapunov equation is solved.
        Only the continuous-time case is implemented.
    options
        The solver options to use (see :func:`lyap_dense_solver_options`).

    Returns
    -------
    X
        Lyapunov equation solution as a |NumPy array|.
    """
    if not cont_time:
        raise NotImplementedError
    _solve_lyap_dense_check_args(A, E, B, trans)
    options = _parse_options(options, lyap_lrcf_solver_options(), 'pymess_glyap', None, False)

    if options['type'] == 'pymess_glyap':
        Y = B.dot(B.T) if not trans else B.T.dot(B)
        op = pymess.MESS_OP_NONE if not trans else pymess.MESS_OP_TRANSPOSE
        X = pymess.glyap(A, E, Y, op=op)[0]
        X = np.asarray(X)
    else:
        raise ValueError(f'Unexpected Lyapunov equation solver ({options["type"]}).')

    return X


@defaults('linesearch', 'maxit', 'absres_tol', 'relres_tol', 'nrm')
def dense_nm_gmpcare_solver_options(linesearch=False,
                                    maxit=50,
                                    absres_tol=1e-11,
                                    relres_tol=1e-12,
                                    nrm=0):
    """Return available Riccati solvers with default options for the pymess backend.

    Also see :func:`lradi_solver_options`.

    Parameters
    ----------
    linesearch
        See `pymess.dense_nm_gmpcare`.
    maxit
        See `pymess.dense_nm_gmpcare`.
    absres_tol
        See `pymess.dense_nm_gmpcare`.
    relres_tol
        See `pymess.dense_nm_gmpcare`.
    nrm
        See `pymess.dense_nm_gmpcare`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'linesearch': linesearch,
            'maxit':      maxit,
            'absres_tol': absres_tol,
            'relres_tol': relres_tol,
            'nrm':        nrm}


@defaults('newton_gstep', 'newton_k0', 'newton_linesearch', 'newton_maxit', 'newton_output', 'newton_res2_tol',
          'newton_singleshifts')
def lrnm_solver_options(newton_gstep=0,
                        newton_k0=None,
                        newton_linesearch=0,
                        newton_maxit=30,
                        newton_output=1,
                        newton_res2_tol=1e-10,
                        newton_singleshifts=0):
    """Return available adi solver options with default values for the pymess backend.

    Parameters
    ----------
    newton_gstep
      See `pymess.OptionsNewton`.
    newton_k0
      See `pymess.OptionsNewton`.
    newton_linesearch
      See `pymess.OptionsNewton`.
    newton_maxit
      See `pymess.OptionsNewton`.
    newton_output
      See `pymess.OptionsNewton`.
    newton_res2_tol
      See `pymess.OptionsNewton`.
    newton_singleshifts
      See `pymess.OptionsNewton`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    lrnm_opts = lradi_solver_options()
    lrnm_opts.nm.gstep = newton_gstep
    lrnm_opts.nm.k0 = newton_k0
    lrnm_opts.nm.linesearch = newton_linesearch
    lrnm_opts.nm.maxit = newton_maxit
    lrnm_opts.nm.output = newton_output
    lrnm_opts.nm.res2_tol = newton_res2_tol
    lrnm_opts.nm.singleshifts = newton_singleshifts

    return lrnm_opts


def ricc_lrcf_solver_options():
    """Return available Riccati solvers with default options for the pymess backend.

    Also see :func:`dense_nm_gmpcare_solver_options` and
    :func:`lrnm_solver_options`.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'pymess_dense_nm_gmpcare': {'type': 'pymess_dense_nm_gmpcare',
                                        'opts': dense_nm_gmpcare_solver_options()},
            'pymess_lrnm':             {'type': 'pymess_lrnm',
                                        'opts': lrnm_solver_options()}}


@defaults('default_solver')
def solve_ricc_lrcf(A, E, B, C, R=None, trans=False, options=None, default_solver=None):
    """Compute an approximate low-rank solution of a Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a
    general description.

    This function uses `pymess.dense_nm_gmpcare` and `pymess.lrnm`.
    For both methods,
    :meth:`~pymor.vectorarrays.interface.VectorArray.to_numpy`
    and
    :meth:`~pymor.vectorarrays.interface.VectorSpace.from_numpy`
    need to be implemented for `A.source`.
    Additionally, since `dense_nm_gmpcare` is a dense solver, it
    expects :func:`~pymor.algorithms.to_matrix.to_matrix` to work
    for A and E.

    If the solver is not specified using the options or
    default_solver arguments, `dense_nm_gmpcare` is used for small
    problems (smaller than defined with
    :func:`~pymor.algorithms.lyapunov.mat_eqn_sparse_min_size`) and
    `lrnm` for large problems.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use (see
        :func:`ricc_lrcf_solver_options`).
    default_solver
        Default solver to use (pymess_lrnm,
        pymess_dense_nm_gmpcare).
        If `None`, chose solver depending on dimension `A`.

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, trans)
    if default_solver is None:
        default_solver = 'pymess_lrnm' if A.source.dim >= mat_eqn_sparse_min_size() else 'pymess_dense_nm_gmpcare'
    options = _parse_options(options, ricc_lrcf_solver_options(), default_solver, None, False)

    if options['type'] == 'pymess_dense_nm_gmpcare':
        X = _call_pymess_dense_nm_gmpare(A, E, B, C, R, trans=trans, options=options['opts'], plus=False,
                                         method_name='solve_ricc_lrcf')
        Z = _chol(X)
    elif options['type'] == 'pymess_lrnm':
        if R is not None:
            import scipy.linalg as spla
            Rc = spla.cholesky(R)                                 # R = Rc^T * Rc
            Rci = spla.solve_triangular(Rc, np.eye(Rc.shape[0]))  # R^{-1} = Rci * Rci^T
            if not trans:
                C = C.lincomb(Rci.T)  # C <- Rci^T * C = (C^T * Rci)^T
            else:
                B = B.lincomb(Rci.T)  # B <- B * Rci
        opts = options['opts']
        opts.type = pymess.MESS_OP_NONE if not trans else pymess.MESS_OP_TRANSPOSE
        eqn = RiccatiEquation(opts, A, E, B, C)
        Z, status = pymess.lrnm(eqn, opts)
        relres = status.res2_norm / status.res2_0
        if relres > opts.adi.res2_tol:
            logger = getLogger('pymor.bindings.pymess.solve_ricc_lrcf')
            logger.warning(f'Desired relative residual tolerance was not achieved '
                           f'({relres:e} > {opts.adi.res2_tol:e}).')
    else:
        raise ValueError(f'Unexpected Riccati equation solver ({options["type"]}).')

    return A.source.from_numpy(Z.T)


def pos_ricc_lrcf_solver_options():
    """Return available positive Riccati solvers with default options for the pymess backend.

    Returns
    -------
    A dict of available solvers with default solver options.
    """
    return {'pymess_dense_nm_gmpcare': {'type': 'pymess_dense_nm_gmpcare',
                                        'opts': dense_nm_gmpcare_solver_options()}}


def solve_pos_ricc_lrcf(A, E, B, C, R=None, trans=False, options=None):
    """Compute an approximate low-rank solution of a positive Riccati equation.

    See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a
    general description.

    This function uses `pymess.dense_nm_gmpcare`.

    Parameters
    ----------
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    R
        The matrix R as a 2D |NumPy array| or `None`.
    trans
        Whether the first |Operator| in the Riccati equation is
        transposed.
    options
        The solver options to use (see
        :func:`pos_ricc_lrcf_solver_options`).

    Returns
    -------
    Z
        Low-rank Cholesky factor of the Riccati equation solution,
        |VectorArray| from `A.source`.
    """
    _solve_ricc_check_args(A, E, B, C, R, trans)
    options = _parse_options(options, pos_ricc_lrcf_solver_options(), 'pymess_dense_nm_gmpcare', None, False)

    if options['type'] == 'pymess_dense_nm_gmpcare':
        X = _call_pymess_dense_nm_gmpare(A, E, B, C, R, trans=trans, options=options['opts'], plus=True,
                                         method_name='solve_pos_ricc_lrcf')
        Z = _chol(X)
    else:
        raise ValueError(f'Unexpected positive Riccati equation solver ({options["type"]}).')

    return A.source.from_numpy(Z.T)


def _call_pymess_dense_nm_gmpare(A, E, B, C, R, trans=False, options=None, plus=False, method_name=''):
    """Return the solution from pymess.dense_nm_gmpare solver."""
    A = to_matrix(A, format='dense')
    E = to_matrix(E, format='dense') if E else None
    B = B.to_numpy().T
    C = C.to_numpy()

    Q = B.dot(B.T) if not trans else C.T.dot(C)
    pymess_trans = pymess.MESS_OP_NONE if not trans else pymess.MESS_OP_TRANSPOSE
    if not trans:
        RinvC = spla.solve(R, C) if R is not None else C
        G = C.T.dot(RinvC)
    else:
        RinvBT = spla.solve(R, B.T) if R is not None else B.T
        G = B.dot(RinvBT)
    X, absres, relres = pymess.dense_nm_gmpare(None,
                                               A, E, Q, G,
                                               plus=plus, trans=pymess_trans,
                                               linesearch=options['linesearch'],
                                               maxit=options['maxit'],
                                               absres_tol=options['absres_tol'],
                                               relres_tol=options['relres_tol'],
                                               nrm=options['nrm'])
    if absres > options['absres_tol']:
        logger = getLogger('pymor.bindings.pymess.' + method_name)
        logger.warning(f'Desired absolute residual tolerance was not achieved '
                       f'({absres:e} > {options["absres_tol"]:e}).')
    if relres > options['relres_tol']:
        logger = getLogger('pymor.bindings.pymess.' + method_name)
        logger.warning(f'Desired relative residual tolerance was not achieved '
                       f'({relres:e} > {options["relres_tol"]:e}).')

    return X


class LyapunovEquation(pymess.Equation):
    """Lyapunov equation class for pymess

    Represents a (generalized) continuous-time algebraic Lyapunov
    equation:

    - if opt.type is `pymess.MESS_OP_NONE` and E is `None`:

      .. math::
          A X + X A^T + B B^T = 0,

    - if opt.type is `pymess.MESS_OP_NONE` and E is not `None`:

      .. math::
          A X E^T + E X A^T + B B^T = 0,

    - if opt.type is `pymess.MESS_OP_TRANSPOSE` and E is `None`:

      .. math::
          A^T X + X A + B^T B = 0,

    - if opt.type is `pymess.MESS_OP_TRANSPOSE` and E is not `None`:

      .. math::
          A^T X E + E^T X A + B^T B = 0.

    Parameters
    ----------
    opt
        pymess Options structure.
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    """

    def __init__(self, opt, A, E, B):
        super().__init__(name='LyapunovEquation', opt=opt, dim=A.source.dim)
        self.a = A
        self.e = E
        self.rhs = B.to_numpy().T
        self.p = []

    def ax_apply(self, op, y):
        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.a.apply(y)
        else:
            x = self.a.apply_adjoint(y)
        return x.to_numpy().T

    def ex_apply(self, op, y):
        if self.e is None:
            return y

        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.e.apply(y)
        else:
            x = self.e.apply_adjoint(y)
        return x.to_numpy().T

    def ainv_apply(self, op, y):
        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.a.apply_inverse(y)
        else:
            x = self.a.apply_inverse_adjoint(y)
        return x.to_numpy().T

    def einv_apply(self, op, y):
        if self.e is None:
            return y

        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.e.apply_inverse(y)
        else:
            x = self.e.apply_inverse_adjoint(y)
        return x.to_numpy().T

    def apex_apply(self, op, p, idx_p, y):
        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.a.apply(y)
            if self.e is None:
                x += p * y
            else:
                x += p * self.e.apply(y)
        else:
            x = self.a.apply_adjoint(y)
            if self.e is None:
                x += p.conjugate() * y
            else:
                x += p.conjugate() * self.e.apply_adjoint(y)
        return x.to_numpy().T

    def apeinv_apply(self, op, p, idx_p, y):
        y = self.a.source.from_numpy(y.T)
        e = IdentityOperator(self.a.source) if self.e is None else self.e

        if p.imag == 0:
            ape = self.a + p.real * e
        else:
            ape = self.a + p * e

        if op == pymess.MESS_OP_NONE:
            x = ape.apply_inverse(y)
        else:
            x = ape.apply_inverse_adjoint(y.conj()).conj()
        return x.to_numpy().T

    def parameter(self, arp_p, arp_m, B=None, K=None):
        return None


class RiccatiEquation(pymess.Equation):
    """Riccati equation class for pymess

    Represents a Riccati equation

    - if opt.type is `pymess.MESS_OP_NONE` and E is `None`:

      .. math::
          A X + X A^T - X C^T C X + B B^T = 0,

    - if opt.type is `pymess.MESS_OP_NONE` and E is not `None`:

      .. math::
          A X E^T + E X A^T - E X C^T C X E^T + B B^T = 0,

    - if opt.type is `pymess.MESS_OP_TRANSPOSE` and E is `None`:

      .. math::
          A^T X + X A - X B B^T X + C^T C = 0,

    - if opt.type is `pymess.MESS_OP_TRANSPOSE` and E is not `None`:

      .. math::
          A^T X E + E^T X A - E X B B^T X E^T + C^T C = 0.

    Parameters
    ----------
    opt
        pymess Options structure.
    A
        The non-parametric |Operator| A.
    E
        The non-parametric |Operator| E or `None`.
    B
        The operator B as a |VectorArray| from `A.source`.
    C
        The operator C as a |VectorArray| from `A.source`.
    """

    def __init__(self, opt, A, E, B, C):
        super().__init__(name='RiccatiEquation', opt=opt, dim=A.source.dim)
        self.a = A
        self.e = E
        self.b = B.to_numpy().T
        self.c = C.to_numpy()
        self.rhs = self.b if opt.type == pymess.MESS_OP_NONE else self.c.T
        self.p = []

    def ax_apply(self, op, y):
        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.a.apply(y)
        else:
            x = self.a.apply_adjoint(y)
        return x.to_numpy().T

    def ex_apply(self, op, y):
        if self.e is None:
            return y

        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.e.apply(y)
        else:
            x = self.e.apply_adjoint(y)
        return x.to_numpy().T

    def ainv_apply(self, op, y):
        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.a.apply_inverse(y)
        else:
            x = self.a.apply_inverse_adjoint(y)
        return x.to_numpy().T

    def einv_apply(self, op, y):
        if self.e is None:
            return y

        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.e.apply_inverse(y)
        else:
            x = self.e.apply_inverse_adjoint(y)
        return x.to_numpy().T

    def apex_apply(self, op, p, idx_p, y):
        y = self.a.source.from_numpy(y.T)
        if op == pymess.MESS_OP_NONE:
            x = self.a.apply(y)
            if self.e is None:
                x += p * y
            else:
                x += p * self.e.apply(y)
        else:
            x = self.a.apply_adjoint(y)
            if self.e is None:
                x += p.conjugate() * y
            else:
                x += p.conjugate() * self.e.apply_adjoint(y)
        return x.to_numpy().T

    def apeinv_apply(self, op, p, idx_p, y):
        y = self.a.source.from_numpy(y.T)
        e = IdentityOperator(self.a.source) if self.e is None else self.e

        if p.imag == 0:
            ape = self.a + p.real * e
        else:
            ape = self.a + p * e

        if op == pymess.MESS_OP_NONE:
            x = ape.apply_inverse(y)
        else:
            x = ape.apply_inverse_adjoint(y.conj()).conj()
        return x.to_numpy().T

    def parameter(self, arp_p, arp_m, B=None, K=None):
        return None
