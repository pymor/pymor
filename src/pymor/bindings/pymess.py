# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_PYMESS:
    import numpy as np
    import scipy.linalg as spla
    import pymess

    from pymor.algorithms.genericsolvers import _parse_options
    from pymor.algorithms.lyapunov import (MAT_EQN_SPARSE_MIN_SIZE, _solve_lyap_lrcf_check_args,
                                           _solve_lyap_dense_check_args, _chol)
    from pymor.algorithms.to_matrix import to_matrix
    from pymor.bindings.scipy import _solve_ricc_check_args
    from pymor.core.defaults import defaults
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

    @defaults('options')
    def solve_lyap_lrcf(A, E, B, trans=False, options=None):
        """Compute an approximate low-rank solution of a Lyapunov equation.

        See :func:`pymor.algorithms.lyapunov.solve_lyap_lrcf` for a
        general description.

        This function uses `pymess.glyap` and `pymess.lradi`.
        For both methods,
        :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.to_numpy`
        and
        :meth:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.from_numpy`
        need to be implemented for `A.source`.
        Additionally, since `glyap` is a dense solver, it expects
        :func:`~pymor.algorithms.to_matrix.to_matrix` to work for A and
        E.

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
            The operator B as a |VectorArray| from `A.source`.
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

        _solve_lyap_lrcf_check_args(A, E, B, trans)
        default_solver = 'pymess_lradi' if A.source.dim >= MAT_EQN_SPARSE_MIN_SIZE else 'pymess_glyap'
        options = _parse_options(options, lyap_lrcf_solver_options(), default_solver, None, False)

        if options['type'] == 'pymess_glyap':
            X = solve_lyap_dense(to_matrix(A, format='dense'),
                                 to_matrix(E, format='dense') if E else None,
                                 B.to_numpy().T if not trans else B.to_numpy(),
                                 trans=trans, options=options)
            Z = _chol(X)
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

        return A.source.from_numpy(Z.T)

    def lyap_dense_solver_options():
        """Returns available Lyapunov equation solvers with default solver options for the pymess backend.

        Returns
        -------
        A dict of available solvers with default solver options.
        """

        return {'pymess_glyap': {'type': 'pymess_glyap'}}

    @defaults('options')
    def solve_lyap_dense(A, E, B, trans=False, options=None):
        """Compute the solution of a Lyapunov equation.

        See :func:`pymor.algorithms.lyapunov.solve_lyap_dense` for a
        general description.

        This function uses `pymess.glyap`.

        Parameters
        ----------
        A
            The operator A as a 2D |NumPy array|.
        E
            The operator E as a 2D |NumPy array| or `None`.
        B
            The operator B as a 2D |NumPy array|.
        trans
            Whether the first operator in the Lyapunov equation is
            transposed.
        options
            The solver options to use (see
            :func:`lyap_dense_solver_options`).

        Returns
        -------
        X
            Lyapunov equation solution as a |NumPy array|.
        """

        _solve_lyap_dense_check_args(A, E, B, trans)
        options = _parse_options(options, lyap_lrcf_solver_options(), 'pymess_glyap', None, False)

        if options['type'] == 'pymess_glyap':
            Y = B.dot(B.T) if not trans else B.T.dot(B)
            op = pymess.MESS_OP_NONE if not trans else pymess.MESS_OP_TRANSPOSE
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

    @defaults('options')
    def solve_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
        """Compute an approximate low-rank solution of a Riccati equation.

        See :func:`pymor.algorithms.riccati.solve_ricc_lrcf` for a
        general description.

        This function uses `pymess.dense_nm_gmpcare` and `pymess.lrnm`.
        For both methods,
        :meth:`~pymor.vectorarrays.interfaces.VectorArrayInterface.to_numpy`
        and
        :meth:`~pymor.vectorarrays.interfaces.VectorSpaceInterface.from_numpy`
        need to be implemented for `A.source`.
        Additionally, since `dense_nm_gmpcare` is a dense solver, it
        expects :func:`~pymor.algorithms.to_matrix.to_matrix` to work
        for A and E.

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
            The operator B as a |VectorArray| from `A.source`.
        C
            The operator C as a |VectorArray| from `A.source`.
        R
            The operator R as a 2D |NumPy array| or `None`.
        S
            The operator S as a |VectorArray| from `A.source` or `None`.
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
            Z = _chol(X)
        elif options['type'] == 'pymess_lrnm':
            if S is not None:
                raise NotImplementedError
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
        else:
            raise ValueError('Unexpected Riccati equation solver ({}).'.format(options['type']))

        return A.source.from_numpy(Z.T)

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

    @defaults('options')
    def solve_pos_ricc_lrcf(A, E, B, C, R=None, S=None, trans=False, options=None):
        """Compute an approximate low-rank solution of a positive Riccati equation.

        See :func:`pymor.algorithms.riccati.solve_pos_ricc_lrcf` for a
        general description.

        This function uses `pymess.dense_nm_gmpcare`.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The operator B as a |VectorArray| from `A.source`.
        C
            The operator C as a |VectorArray| from `A.source`.
        R
            The operator R as a 2D |NumPy array| or `None`.
        S
            The operator S as a |VectorArray| from `A.source` or `None`.
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
            Z = _chol(X)
        else:
            raise ValueError('Unexpected positive Riccati equation solver ({}).'.format(options['type']))

        return A.source.from_numpy(Z.T)

    def _call_pymess_dense_nm_gmpare(A, E, B, C, R, S, trans=False, options=None, plus=False):
        """Return the solution from pymess.dense_nm_gmpare solver."""
        A = to_matrix(A, format='dense')
        E = to_matrix(E, format='dense') if E else None
        B = B.to_numpy().T
        C = C.to_numpy()
        S = S.to_numpy().T if S else None

        Q = B.dot(B.T) if not trans else C.T.dot(C)
        pymess_trans = pymess.MESS_OP_NONE if not trans else pymess.MESS_OP_TRANSPOSE
        if not trans:
            RinvC = spla.solve(R, C) if R is not None else C
            G = C.T.dot(RinvC)
            if S is not None:
                RinvST = spla.solve(R, S.T) if R is not None else S.T
                if not plus:
                    A -= S.dot(RinvC)
                    Q -= S.dot(RinvST)
                else:
                    A += S.dot(RinvC)
                    Q += S.dot(RinvST)
        else:
            RinvBT = spla.solve(R, B.T) if R is not None else B.T
            G = B.dot(RinvBT)
            if S is not None:
                RinvST = spla.solve(R, S.T) if R is not None else S.T
                if not plus:
                    A -= RinvBT.T.dot(S.T)
                    Q -= S.dot(RinvST)
                else:
                    A += RinvBT.T.dot(S.T)
                    Q += S.dot(RinvST)
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
            The |Operator| A.
        E
            The |Operator| E or `None`.
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
        r"""Riccati equation class for pymess

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
            The |Operator| A.
        E
            The |Operator| E or `None`.
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
