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
    from pymor.core.defaults import defaults
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

    def ricc_solver_options():
        """Returns available Riccati equation solvers with default |solver_options| for the pymess backend.

        Returns
        -------
        A dict of available solvers with default |solver_options|.
        """
        opts = pymess.Options()
        opts.adi.shifts.paratype = pymess.MESS_LRCFADI_PARA_ADAPTIVE_V

        return {'pymess':      {'type': 'pymess',
                                'opts': opts},
                'pymess_care': {'type': 'pymess_care'},
                'pymess_lrnm': {'type': 'pymess_lrnm',
                                'opts': opts}}

    @defaults('default_solver')
    def solve_ricc(A, E=None, B=None, Q=None, C=None, R=None, G=None,
                   trans=False, options=None, default_solver='pymess'):
        """Find a factor of the solution of a Riccati equation

        Returns factor :math:`Z` such that :math:`Z Z^T` is
        approximately the solution :math:`X` of a Riccati equation

        .. math::
            A^T X E + E^T X A - E^T X B R^{-1} B^T X E + Q = 0.

        If E in `None`, it is taken to be the identity matrix.
        Q can instead be given as C^T * C. In this case, Q needs to be
        `None`, and C not `None`.
        B * R^{-1} B^T can instead be given by G. In this case, B and R
        need to be `None`, and G not `None`.
        If R and G are `None`, then R is taken to be the identity
        matrix.
        If trans is `True`, then the dual Riccati equation is solved

        .. math::
            A X E^T + E X A^T - E X C^T R^{-1} C X E^T + Q = 0,

        where Q can be replaced by B * B^T and C^T * R^{-1} * C by G.

        This uses the `pymess` package, in particular its `care` and
        `lrnm` methods.
        Operators Q, R, and G are not supported,
        Both methods can be used for large-scale problems.
        The restrictions are:

            - `care` needs access to all matrix data, i.e., it expects
              :func:`~pymor.algorithms.to_matrix.to_matrix` to work for
              A, E, B, and C,
            - `lrnm` needs access to the data of the operators B and C,
              i.e., it expects
              :func:`~pymor.algorithms.to_matrix.to_matrix` to work for
              B and C.

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
        G
            The |Operator| G or `None`.
        trans
            If the dual equation needs to be solved.
        options
            The |solver_options| to use (see
            :func:`ricc_solver_options`).
        default_solver
            The solver to use when no `options` are specified (pymess,
            pymess_care, pymess_lrnm).

        Returns
        -------
        Z
            Low-rank factor of the Riccati equation solution,
            |VectorArray| from `A.source`.
        """
        _solve_ricc_check_args(A, E, B, Q, C, R, G, trans)
        options = _parse_options(options, ricc_solver_options(), default_solver, None, False)

        if options['type'] == 'pymess':
            if A.source.dim >= MAT_EQN_SPARSE_MIN_SIZE:
                options = dict(options, type='pymess_lrnm')  # do not modify original dict!
            else:
                options = dict(options, type='pymess_care')  # do not modify original dict!

        if options['type'] == 'pymess_care':
            if Q is not None or R is not None or G is not None:
                raise NotImplementedError
            A_mat = to_matrix(A, format='dense') if A.source.dim < MAT_EQN_SPARSE_MIN_SIZE else to_matrix(A)
            if E is not None:
                E_mat = to_matrix(E, format='dense') if A.source.dim < MAT_EQN_SPARSE_MIN_SIZE else to_matrix(E)
            else:
                E_mat = None
            B_mat = to_matrix(B, format='dense') if B else None
            C_mat = to_matrix(C, format='dense') if C else None
            if not trans:
                Z = pymess.care(A_mat, E_mat, B_mat, C_mat)
            else:
                if E is None:
                    Z = pymess.care(A_mat.T, None, C_mat.T, B_mat.T)
                else:
                    Z = pymess.care(A_mat.T, E_mat.T, C_mat.T, B_mat.T)
        elif options['type'] == 'pymess_lrnm':
            if Q is not None or R is not None or G is not None:
                raise NotImplementedError
            opts = options['opts']
            if not trans:
                opts.type = pymess.MESS_OP_TRANSPOSE
            else:
                opts.type = pymess.MESS_OP_NONE
            eqn = RiccatiEquation(opts, A, E, B, C)
            Z, status = pymess.lrnm(eqn, opts)

        Z = A.source.from_numpy(np.array(Z).T)

        return Z

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
