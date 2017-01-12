# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_PYMESS:
    import numpy as np
    import pymess

    from pymor.algorithms.to_matrix import to_matrix
    from pymor.bindings.scipy import _solve_lyap_check_args, _solve_ricc_check_args
    from pymor.operators.constructions import IdentityOperator, LincombOperator


    def solve_lyap(A, E, B, trans=False, me_solver='pymess', tol=None):
        """Find a factor of the solution of a Lyapunov equation.

        Returns factor :math:`Z` such that :math:`Z Z^T` is approximately
        the solution :math:`X` of a Lyapunov equation (if E is `None`).

        .. math::
            A X + X A^T + B B^T = 0

        or generalized Lyapunov equation

        .. math::
            A X E^T + E X A^T + B B^T = 0.

        If trans is `True`, then solve (if E is `None`)

        .. math::
            A^T X + X A + B^T B = 0

        or

        .. math::
            A^T X E + E^T X A + B^T B = 0.

        Parameters
        ----------
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        trans
            If the dual equation needs to be solved.
        me_solver
            Solver to use ('pymess', 'pymess_lyap', 'pymess_lradi').
            If `me_solver` is `'pymess'`, the specific solver is chosen
            depending on the size of the problem.
        tol
            Tolerance parameter for pymess_lradi solver.

        Returns
        -------
        Z
            Low-rank factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
        """
        _solve_lyap_check_args(A, E, B, trans)
        assert me_solver in ('pymess', 'pymess_lyap', 'pymess_lradi')

        if me_solver == 'pymess':
            if A.source.dim >= 1000:
                me_solver = 'pymess_lradi'
            else:
                me_solver = 'pymess_lyap'

        if me_solver == 'pymess_lyap':
            A_mat = to_matrix(A) if A.source.dim < 1000 else to_matrix(A, format='csc')
            if E is not None:
                E_mat = to_matrix(E) if E.source.dim < 1000 else to_matrix(E, format='csc')
            B_mat = to_matrix(B)
            if not trans:
                if E is None:
                    Z = pymess.lyap(A_mat, None, B_mat)
                else:
                    Z = pymess.lyap(A_mat, E_mat, B_mat)
            else:
                if E is None:
                    Z = pymess.lyap(A_mat.T, None, B_mat.T)
                else:
                    Z = pymess.lyap(A_mat.T, E_mat.T, B_mat.T)
        elif me_solver == 'pymess_lradi':
            opts = pymess.options()
            opts.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V
            if trans:
                opts.type = pymess.MESS_OP_TRANSPOSE
            if tol is not None:
                opts.rel_change_tol = tol
                opts.adi.res2_tol = tol
                opts.adi.res2c_tol = tol
            eqn = LyapunovEquation(opts, A, E, B)
            Z, status = pymess.lradi(eqn, opts)

        Z = A.source.from_data(np.array(Z).T)

        return Z


    def solve_ricc(A, E=None, B=None, Q=None, C=None, R=None, G=None,
                   trans=False, me_solver='pymess', tol=None):
        """Find a factor of the solution of a Riccati equation

        Returns factor :math:`Z` such that :math:`Z Z^T` is approximately the
        solution :math:`X` of a Riccati equation

        .. math::
            A^T X E + E^T X A - E^T X B R^{-1} B^T X E + Q = 0.

        If E in `None`, it is taken to be the identity matrix.
        Q can instead be given as C^T * C. In this case, Q needs to be `None`, and
        C not `None`.
        B * R^{-1} B^T can instead be given by G. In this case, B and R need to be
        `None`, and G not `None`.
        If R and G are `None`, then R is taken to be the identity matrix.
        If trans is `True`, then the dual Riccati equation is solved

        .. math::
            A X E^T + E X A^T - E X C^T R^{-1} C X E^T + Q = 0,

        where Q can be replaced by B * B^T and C^T * R^{-1} * C by G.

        Parameters
        ----------
        A
            The |Operator| A.
        B
            The |Operator| B or `None`.
        E
            The |Operator| E or `None`.
        Q
            The |Operator| Q or `None`.
        C
            The |Operator| C or `None`.
        R
            The |Operator| R or `None`.
        D
            The |Operator| D or `None`.
        G
            The |Operator| G or `None`.
        L
            The |Operator| L or `None`.
        trans
            If the dual equation needs to be solved.
        me_solver
            Method to use ('pymess', 'pymess_care', 'pymess_lrnm').
            If `me_solver` is `'pymess'`, the specific solver is chosen
            depending on the size of the problem.
        tol
            Tolerance parameter for pymess_lrnm solver.

        Returns
        -------
        Z
            Low-rank factor of the Riccati equation solution,
            |VectorArray| from `A.source`.
        """
        _solve_ricc_check_args(A, E, B, Q, C, R, G, trans)
        assert me_solver in {'pymess', 'pymess_care', 'pymess_lrnm'}

        if me_solver == 'pymess':
            if A.source.dim >= 1000:
                me_solver = 'pymess_lrnm'
            else:
                me_solver = 'pymess_care'

        if me_solver == 'pymess_care':
            if Q is not None or R is not None or G is not None:
                raise NotImplementedError()
            A_mat = to_matrix(A)
            E_mat = to_matrix(E) if E else None
            B_mat = to_matrix(B) if B else None
            C_mat = to_matrix(C) if C else None
            if not trans:
                Z = pymess.care(A_mat, E_mat, B_mat, C_mat)
            else:
                if E is None:
                    Z = pymess.care(A_mat.T, None, C_mat.T, B_mat.T)
                else:
                    Z = pymess.care(A_mat.T, E_mat.T, C_mat.T, B_mat.T)
        elif me_solver == 'pymess_lrnm':
            if Q is not None or R is not None or G is not None:
                raise NotImplementedError()
            opts = pymess.options()
            opts.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V
            if not trans:
                opts.type = pymess.MESS_OP_TRANSPOSE
            if tol is not None:
                opts.rel_change_tol = tol
                opts.adi.res2_tol = tol
                opts.adi.res2c_tol = tol
            eqn = RiccatiEquation(opts, A, E, B, C)
            Z, status = pymess.lrnm(eqn, opts)

        Z = A.source.from_data(np.array(Z).T)

        return Z


    class LyapunovEquation(pymess.equation):
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
            pymess options structure.
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        B
            The |Operator| B.
        """
        def __init__(self, opt, A, E, B):
            super().__init__(name='lyap_eqn', opt=opt, dim=A.source.dim)

            self.A = A
            self.E = E
            self.RHS = to_matrix(B)
            if opt.type == pymess.MESS_OP_TRANSPOSE:
                self.RHS = self.RHS.T
            self.p = []

        def AX_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
            else:
                x = self.A.apply_transpose(y)
            return np.matrix(x.data).T

        def EX_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply(y)
            else:
                x = self.E.apply_transpose(y)
            return np.matrix(x.data).T

        def AINV_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply_inverse(y)
            else:
                x = self.A.apply_inverse_transpose(y)
            return np.matrix(x.data).T

        def EINV_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply_inverse(y)
            else:
                x = self.E.apply_inverse_transpose(y)
            return np.matrix(x.data).T

        def ApEX_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
                if self.E is None:
                    x += p * y
                else:
                    x += p * self.E.apply(y)
            else:
                x = self.A.apply_transpose(y)
                if self.E is None:
                    x += p.conjugate() * y
                else:
                    x += p.conjugate() * self.E.apply_transpose(y)
            return np.matrix(x.data).T

        def ApEINV_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            E = IdentityOperator(self.A.source) if self.E is None else self.E

            if p.imag == 0:
                ApE = LincombOperator((self.A, E), (1, p.real))
            else:
                ApE = LincombOperator((self.A, E), (1, p))

            if op == pymess.MESS_OP_NONE:
                x = ApE.apply_inverse(y)
            else:
                x = ApE.apply_inverse_transpose(y)
            return np.matrix(x.data).T

        def parameter(self, arp_p, arp_m, B=None, K=None):
            return None


    class RiccatiEquation(pymess.equation):
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
            pymess options structure.
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
            super().__init__(name='ricc_eqn', opt=opt, dim=A.source.dim)

            self.A = A
            self.E = E
            self.B = to_matrix(B)
            self.C = to_matrix(C)
            self.RHS = self.B if opt.type == pymess.MESS_OP_NONE else self.C.T
            self.p = []

        def AX_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
            else:
                x = self.A.apply_transpose(y)
            return np.matrix(x.data).T

        def EX_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply(y)
            else:
                x = self.E.apply_transpose(y)
            return np.matrix(x.data).T

        def AINV_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply_inverse(y)
            else:
                x = self.A.apply_inverse_transpose(y)
            return np.matrix(x.data).T

        def EINV_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply_inverse(y)
            else:
                x = self.E.apply_inverse_transpose(y)
            return np.matrix(x.data).T

        def ApEX_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
                if self.E is None:
                    x += p * y
                else:
                    x += p * self.E.apply(y)
            else:
                x = self.A.apply_transpose(y)
                if self.E is None:
                    x += p.conjugate() * y
                else:
                    x += p.conjugate() * self.E.apply_transpose(y)
            return np.matrix(x.data).T

        def ApEINV_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            E = IdentityOperator(self.A.source) if self.E is None else self.E

            if p.imag == 0:
                ApE = LincombOperator((self.A, E), (1, p.real))
            else:
                ApE = LincombOperator((self.A, E), (1, p))

            if op == pymess.MESS_OP_NONE:
                x = ApE.apply_inverse(y)
            else:
                x = ApE.apply_inverse_transpose(y)
            return np.matrix(x.data).T

        def parameter(self, arp_p, arp_m, B=None, K=None):
            return None
