# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_PYMESS:
    import numpy as np
    import pymess

    from pymor.algorithms.genericsolvers import _parse_options
    from pymor.algorithms.lyapunov import MAT_EQN_SPARSE_MIN_SIZE, _solve_lyap_check_args, chol
    from pymor.algorithms.to_matrix import to_matrix
    from pymor.bindings.scipy import _solve_ricc_check_args
    from pymor.core.logger import getLogger
    from pymor.operators.constructions import IdentityOperator

    def lyap_lrcf_solver_options(lradi_opts=None):
        """Returns available Lyapunov equation solvers with default solver options for the pymess backend.

        Parameters
        ----------
        lradi_opts
            Options for `pymess.lradi` (see `pymess.Options()`).

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        if lradi_opts is None:
            lradi_opts = pymess.Options()
            lradi_opts.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V

        return {'pymess_glyap': {'type': 'pymess_glyap'},
                'pymess_lradi': {'type': 'pymess_lradi',
                                 'opts': lradi_opts}}

    def solve_lyap_lrcf(A, E, B, trans=False, options=None):
        """Compute an approximate low-rank solution of a Lyapunov equation.

        See :func:`pymor.algorithms.lyapunov.solve_lyap_lrcf` for a
        general description.

        This function uses `pymess`, in particular its `glyap` and
        `lradi` methods:

        - `glyap` is a dense solver and expects
          :func:`~pymor.algorithms.to_matrix.to_matrix` to work for A,
          E, and B,
        - `lradi` is a sparse solver and expects
          :func:`~pymor.algorithms.to_matrix.to_matrix` to work for B
          and
          :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.to_numpy`
          and
          :meth:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.from_numpy`
          to be implemented for `A.source`.

        If the solver is not specified using the options argument,
        `glyap` is used for small problems (smaller than
        `MAT_EQN_SPARSE_MIN_SIZE`) and `lradi` for large problems.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        trans
            Whether the first |Operator| in the Lyapunov equation is
            transposed.
        options
            The solver options to use (see
            :func:`lyap_lrcf_solver_options`).

        Returns
        -------
        Z
            Low-rank Cholesky factor of the Lyapunov equation solution,
            |VectorArray| from `A.source`.
        """

        _solve_lyap_check_args(A, E, B, trans)
        default_solver = 'pymess_lradi' if A.source.dim >= MAT_EQN_SPARSE_MIN_SIZE else 'pymess_glyap'
        options = _parse_options(options, lyap_lrcf_solver_options(), default_solver, None, False)

        if options['type'] == 'pymess_glyap':
            X = solve_lyap_dense(A, E, B, trans=trans)
            Z = chol(X)
        elif options['type'] == 'pymess_lradi':
            opts = options['opts']
            if not trans:
                opts.type = pymess.MESS_OP_NONE
            else:
                opts.type = pymess.MESS_OP_TRANSPOSE
            eqn = LyapunovEquation(opts, A, E, B)
            Z, status = pymess.lradi(eqn, opts)
        else:
            raise ValueError('Unexpected Lyapunov equation solver ({}).'.format(options['type']))

        Z = A.source.from_numpy(np.array(Z).T)
        return Z

    def lyap_dense_solver_options():
        """Returns available Lyapunov equation solvers with default solver options for the pymess backend.

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        return {'pymess_glyap': {'type': 'pymess_glyap'}}

    def solve_lyap_dense(A, E, B, trans=False, options=None):
        """Compute the solution of a Lyapunov equation.

        See :func:`pymor.algorithms.lyapunov.solve_lyap_dense` for a
        general description.

        This function uses `pymess.glyap`.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        trans
            Whether the first |Operator| in the Lyapunov equation is
            transposed.
        options
            The solver options to use (see
            :func:`lyap_dense_solver_options`).

        Returns
        -------
        X
            Lyapunov equation solution as a |NumPy array|.
        """

        _solve_lyap_check_args(A, E, B, trans)
        options = _parse_options(options, lyap_lrcf_solver_options(), 'pymess_glyap', None, False)

        if options['type'] == 'pymess_glyap':
            A = to_matrix(A, format='dense')
            if E is not None:
                E = to_matrix(E, format='dense')
            B = to_matrix(B, format='dense')
            if not trans:
                Y = B.dot(B.T)
                op = pymess.MESS_OP_NONE
            else:
                Y = B.T.dot(B)
                op = pymess.MESS_OP_TRANSPOSE
            X = pymess.glyap(A, E, Y, op=op)[0]
        else:
            raise ValueError('Unexpected Lyapunov equation solver ({}).'.format(options['type']))

        return X

    def ricc_lrcf_solver_options(dense_nm_gmpcare_linesearch=False,
                                 dense_nm_gmpcare_maxit=50,
                                 dense_nm_gmpcare_absres_tol=1e-11,
                                 dense_nm_gmpcare_relres_tol=1e-12,
                                 dense_nm_gmpcare_nrm=0,
                                 lrnm_opts=None):
        """Returns available Riccati equation solvers with default solver options for the pymess backend.

        Parameters
        ----------
        dense_nm_gmpcare_linesearch
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_maxit
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_absres_tol
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_relres_tol
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_nrm
            See `pymess.dense_nm_gmpcare`.
        lrnm_opts
            Options for `pymess.lrnm` (see `pymess.Options()`).

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        if lrnm_opts is None:
            lrnm_opts = pymess.Options()
            lrnm_opts.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V

        return {'pymess_dense_nm_gmpcare': {'type': 'pymess_dense_nm_gmpcare',
                                            'linesearch': dense_nm_gmpcare_linesearch,
                                            'maxit': dense_nm_gmpcare_maxit,
                                            'absres_tol': dense_nm_gmpcare_absres_tol,
                                            'relres_tol': dense_nm_gmpcare_relres_tol,
                                            'nrm': dense_nm_gmpcare_nrm},
                'pymess_lrnm':             {'type': 'pymess_lrnm',
                                            'opts': lrnm_opts}}

    def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
        """Compute an approximate low-rank solution of a Riccati equation.

        See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a
        general description.

        This function uses `pymess`, in particular its
        `dense_nm_gmpcare` and `lrnm` methods:

        - `dense_nm_gmpcare` is a dense solver and expects
          :func:`~pymor.algorithms.to_matrix.to_matrix` to work for all
          |Operators|,
        - `lrnm` is a sparse solver and expects
          :func:`~pymor.algorithms.to_matrix.to_matrix` to work for B
          and C, and
          :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.to_numpy`
          and
          :meth:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.from_numpy`
          to be implemented for `A.source`.

        If the solver is not specified using the options argument,
        `dense_nm_gmpcare` is used for small problems (smaller than
        `MAT_EQN_SPARSE_MIN_SIZE`) and `lrnm` for large problems.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        C
            The |Operator| C.
        R
            The |Operator| R or `None`.
        S
            The |Operator| S or `None`.
        trans
            Whether the first |Operator| in the Riccati equation is
            transposed.
        options
            The solver options to use (see :func:`ricc_solver_options`).

        Returns
        -------
        Z
            Low-rank Cholesky factor of the Riccati equation solution,
            |VectorArray| from `A.source`.
        """

        _solve_ricc_check_args(A, E, B, C, R, S, trans)
        default_solver = 'pymess_lrnm' if A.source.dim >= MAT_EQN_SPARSE_MIN_SIZE else 'pymess_dense_nm_gmpcare'
        options = _parse_options(options, ricc_lrcf_solver_options(), default_solver, None, False)

        if options['type'] == 'pymess_dense_nm_gmpcare':
            X = _call_pymess_dense_nm_gmpare(A, E, B, C, R, S, trans=trans, options=options, plus=False)
            Z = chol(X)
        elif options['type'] == 'pymess_lrnm':
            if R is not None and S is not None:
                from pymor.operators.constructions import InverseOperator
                if not trans:
                    A = A - S @ InverseOperator(R) @ C
                else:
                    A = A - B @ InverseOperator(R) @ S.H
            elif S is not None:
                if not trans:
                    A = A - S @ C
                else:
                    A = A - B @ S.H
            if R is not None:
                import scipy.linalg as spla
                from pymor.operators.constructions import VectorArrayOperator
                if not trans:
                    R_chol = spla.cholesky(to_matrix(R, format='dense'),
                                           lower=True)
                    R_chol_inv = spla.solve_triangular(R_chol, np.eye(R_chol.shape[0]),
                                                       lower=True)
                    R_chol_inv_va = C.range.from_numpy(R_chol_inv.T)
                    R_chol_inv_va_op = VectorArrayOperator(R_chol_inv_va)
                    C = R_chol_inv_va_op @ C
                else:
                    R_chol = spla.cholesky(to_matrix(R, format='dense'))
                    R_chol_inv = spla.solve_triangular(R_chol, np.eye(R_chol.shape[0]))
                    R_chol_inv_va = B.source.from_numpy(R_chol_inv.T)
                    R_chol_inv_va_op = VectorArrayOperator(R_chol_inv_va)
                    B = B @ R_chol_inv_va_op
            opts = options['opts']
            if not trans:
                opts.type = pymess.MESS_OP_NONE
            else:
                opts.type = pymess.MESS_OP_TRANSPOSE
            eqn = RiccatiEquation(opts, A, E, B, C)
            Z, status = pymess.lrnm(eqn, opts)
        else:
            raise ValueError('Unexpected Riccati equation solver ({}).'.format(options['type']))

        Z = A.source.from_numpy(np.array(Z).T)

        return Z

    def pos_ricc_lrcf_solver_options(dense_nm_gmpcare_linesearch=False,
                                     dense_nm_gmpcare_maxit=50,
                                     dense_nm_gmpcare_absres_tol=1e-11,
                                     dense_nm_gmpcare_relres_tol=1e-12,
                                     dense_nm_gmpcare_nrm=0):
        """Returns available positive Riccati equation solvers with default solver options for the pymess backend.

        Parameters
        ----------
        dense_nm_gmpcare_linesearch
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_maxit
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_absres_tol
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_relres_tol
            See `pymess.dense_nm_gmpcare`.
        dense_nm_gmpcare_nrm
            See `pymess.dense_nm_gmpcare`.

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        return {'pymess_dense_nm_gmpcare': {'type': 'pymess_dense_nm_gmpcare',
                                            'linesearch': dense_nm_gmpcare_linesearch,
                                            'maxit': dense_nm_gmpcare_maxit,
                                            'absres_tol': dense_nm_gmpcare_absres_tol,
                                            'relres_tol': dense_nm_gmpcare_relres_tol,
                                            'nrm': dense_nm_gmpcare_nrm}}

    def solve_pos_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
        """Compute an approximate low-rank solution of a positive Riccati equation.

        See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a
        general description.

        This function uses `pymess.dense_nm_gmpcare`, which is a dense
        solver and expects :func:`~pymor.algorithms.to_matrix.to_matrix`
        to work for all |Operators|,

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        C
            The |Operator| C.
        R
            The |Operator| R or `None`.
        S
            The |Operator| S or `None`.
        trans
            Whether the first |Operator| in the Riccati equation is
            transposed.
        options
            The solver options to use (see :func:`ricc_solver_options`).

        Returns
        -------
        Z
            Low-rank Cholesky factor of the Riccati equation solution,
            |VectorArray| from `A.source`.
        """

        _solve_ricc_check_args(A, E, B, C, R, S, trans)
        options = _parse_options(options, pos_ricc_lrcf_solver_options(), 'pymess_dense_nm_gmpcare', None, False)

        if options['type'] == 'pymess_dense_nm_gmpcare':
            X = _call_pymess_dense_nm_gmpare(A, E, B, C, R, S, trans=trans, options=options, plus=True)
            Z = chol(X)
        else:
            raise ValueError('Unexpected positive Riccati equation solver ({}).'.format(options['type']))

        Z = A.source.from_numpy(np.array(Z).T)

        return Z

    def _call_pymess_dense_nm_gmpare(A, E, B, C, R, S, trans=False, options=None, plus=False):
        """Return the solution from pymess.dense_nm_gmpare solver."""
        A = to_matrix(A, format='dense')
        E = to_matrix(E, format='dense') if E else None
        B = to_matrix(B, format='dense')
        C = to_matrix(C, format='dense')
        R = to_matrix(R, format='dense') if R else None
        S = to_matrix(S, format='dense') if S else None
        if not trans:
            Q = B.dot(B.T)
            if R is None:
                G = C.T.dot(C)
                if S is not None:
                    if not plus:
                        A -= S.dot(C)
                        Q -= S.dot(S.T)
                    else:
                        A += S.dot(C)
                        Q += S.dot(S.T)
            else:
                import scipy.linalg as spla
                RinvC = spla.solve(R, C)
                G = C.T.dot(RinvC)
                if S is not None:
                    if not plus:
                        A -= S.dot(RinvC)
                        Q -= S.dot(spla.solve(R, S.T))
                    else:
                        A += S.dot(RinvC)
                        Q += S.dot(spla.solve(R, S.T))
            pymess_trans = pymess.MESS_OP_NONE
        else:
            Q = C.T.dot(C)
            if R is None:
                G = B.dot(B.T)
                if S is not None:
                    if not plus:
                        A -= B.dot(S.T)
                        Q -= S.dot(S.T)
                    else:
                        A += B.dot(S.T)
                        Q += S.dot(S.T)
            else:
                import scipy.linalg as spla
                RinvBT = spla.solve(R, B.T)
                G = B.dot(RinvBT)
                if S is not None:
                    if not plus:
                        A -= RinvBT.T.dot(S.T)
                        Q -= S.dot(spla.solve(R, S.T))
                    else:
                        A += RinvBT.T.dot(S.T)
                        Q += S.dot(spla.solve(R, S.T))
            pymess_trans = pymess.MESS_OP_TRANSPOSE
        X, absres, relres = pymess.dense_nm_gmpare(None,
                                                   A, E, Q, G,
                                                   plus=plus, trans=pymess_trans,
                                                   linesearch=options['linesearch'],
                                                   maxit=options['maxit'],
                                                   absres_tol=options['absres_tol'],
                                                   relres_tol=options['relres_tol'],
                                                   nrm=options['nrm'])
        if absres > options['absres_tol']:
            logger = getLogger('pymess.dense_nm_gmpcare')
            logger.warning('Desired absolute residual tolerance was not achieved '
                           '({:e} > {:e}).'.format(absres, options['absres_tol']))
        if relres > options['relres_tol']:
            logger = getLogger('pymess.dense_nm_gmpcare')
            logger.warning('Desired relative residual tolerance was not achieved '
                           '({:e} > {:e}).'.format(relres, options['relres_tol']))

        return X

    class LyapunovEquation(pymess.Equation):
        r"""Lyapunov equation class for pymess

        Represents a Lyapunov equation

        .. math::
            A X + X A^T + B B^T = 0

        if E is `None`, otherwise a generalized Lyapunov equation

        .. math::
            A X E^T + E X A^T + B B^T = 0.

        For the dual Lyapunov equation

        .. math::
            A^T X + X A + B^T B = 0, \\
            A^T X E + E^T X A + B^T B = 0,

        `opt.type` needs to be `pymess.MESS_OP_TRANSPOSE`.

        Parameters
        ----------
        opt
            pymess Options structure.
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        """
        def __init__(self, opt, A, E, B):
            super().__init__(name='LyapunovEquation', opt=opt, dim=A.source.dim)

            self.a = A
            self.e = E
            self.rhs = to_matrix(B, format='dense')
            if opt.type == pymess.MESS_OP_TRANSPOSE:
                self.rhs = self.rhs.T
            self.p = []

        def ax_apply(self, op, y):
            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.a.apply(y)
            else:
                x = self.a.apply_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def ex_apply(self, op, y):
            if self.e is None:
                return y

            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.e.apply(y)
            else:
                x = self.e.apply_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def ainv_apply(self, op, y):
            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.a.apply_inverse(y)
            else:
                x = self.a.apply_inverse_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def einv_apply(self, op, y):
            if self.e is None:
                return y

            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.e.apply_inverse(y)
            else:
                x = self.e.apply_inverse_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def apex_apply(self, op, p, idx_p, y):
            y = self.a.source.from_numpy(np.array(y).T)
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
            return np.matrix(x.to_numpy()).T

        def apeinv_apply(self, op, p, idx_p, y):
            y = self.a.source.from_numpy(np.array(y).T)
            e = IdentityOperator(self.a.source) if self.e is None else self.e

            if p.imag == 0:
                ape = self.a + p.real * e
            else:
                ape = self.a + p * e

            if op == pymess.MESS_OP_NONE:
                x = ape.apply_inverse(y)
            else:
                x = ape.apply_inverse_adjoint(y.conj()).conj()
            return np.matrix(x.to_numpy()).T

        def parameter(self, arp_p, arp_m, B=None, K=None):
            return None

    class RiccatiEquation(pymess.Equation):
        r"""Riccati equation class for pymess

        Represents a Riccati equation

        .. math::
            A^T X + X A - X B B^T X + C^T C = 0

        if E is `None`, otherwise a generalized Lyapunov equation

        .. math::
            A^T X E + E^T X A - E^T X B B^T X E + C^T C = 0.

        For the dual Riccati equation

        .. math::
            A X + X A^T - X C^T C X + B B^T = 0, \\
            A X E^T + E X A^T - E X C^T C X E^T + B B^T = 0,

        `opt.type` needs to be `pymess.MESS_OP_NONE`.

        Parameters
        ----------
        opt
            pymess Options structure.
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        C
            The |Operator| C.
        """
        def __init__(self, opt, A, E, B, C):
            super().__init__(name='RiccatiEquation', opt=opt, dim=A.source.dim)

            self.a = A
            self.e = E
            self.b = to_matrix(B, format='dense')
            self.c = to_matrix(C, format='dense')
            self.rhs = self.b if opt.type == pymess.MESS_OP_NONE else self.c.T
            self.p = []

        def ax_apply(self, op, y):
            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.a.apply(y)
            else:
                x = self.a.apply_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def ex_apply(self, op, y):
            if self.e is None:
                return y

            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.e.apply(y)
            else:
                x = self.e.apply_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def ainv_apply(self, op, y):
            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.a.apply_inverse(y)
            else:
                x = self.a.apply_inverse_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def einv_apply(self, op, y):
            if self.e is None:
                return y

            y = self.a.source.from_numpy(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.e.apply_inverse(y)
            else:
                x = self.e.apply_inverse_adjoint(y)
            return np.matrix(x.to_numpy()).T

        def apex_apply(self, op, p, idx_p, y):
            y = self.a.source.from_numpy(np.array(y).T)
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
            return np.matrix(x.to_numpy()).T

        def apeinv_apply(self, op, p, idx_p, y):
            y = self.a.source.from_numpy(np.array(y).T)
            e = IdentityOperator(self.a.source) if self.e is None else self.e

            if p.imag == 0:
                ape = self.a + p.real * e
            else:
                ape = self.a + p * e

            if op == pymess.MESS_OP_NONE:
                x = ape.apply_inverse(y)
            else:
                x = ape.apply_inverse_adjoint(y.conj()).conj()
            return np.matrix(x.to_numpy()).T

        def parameter(self, arp_p, arp_m, B=None, K=None):
            return None
