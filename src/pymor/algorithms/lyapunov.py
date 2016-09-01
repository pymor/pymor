# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse.linalg as spsla

from pymor.algorithms.to_matrix import to_matrix
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.constructions import IdentityOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator


try:
    import pymess

    class LyapunovEquation(pymess.equation):
        r"""Lyapunov equation class for pymess

        Represents a Lyapunov equation

        .. math::
            A X + X A^T + \text{RHS} \cdot \text{RHS}^T = 0

        if E is `None`, otherwise a generalized Lyapunov equation

        .. math::
            A X E^T + E X A^T + \text{RHS} \cdot \text{RHS}^T = 0.

        For the dual Lyapunov equation

        .. math::
            A^T X + X A + \text{RHS}^T \cdot \text{RHS} = 0, \\
            A^T X E + E^T X A + \text{RHS}^T \cdot \text{RHS} = 0,

        `opt.type` needs to be `pymess.MESS_OP_TRANSPOSE`.

        Parameters
        ----------
        opt
            pymess options structure.
        A
            The |Operator| A.
        E
            The |Operator| E or `None`.
        RHS
            The |Operator| RHS.
        """
        def __init__(self, opt, A, E, RHS):
            super(LyapunovEquation, self).__init__(name='lyap_eqn', opt=opt, dim=A.source.dim)

            self.A = A
            self.E = E
            self.RHS = to_matrix(RHS)
            if opt.type == pymess.MESS_OP_TRANSPOSE:
                self.RHS = self.RHS.T
            self.p = []

        def A_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
            else:
                x = self.A.apply_adjoint(y)
            return np.matrix(x.data).T

        def E_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply(y)
            else:
                x = self.E.apply_adjoint(y)
            return np.matrix(x.data).T

        def As_apply(self, op, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply_inverse(y)
            else:
                x = self.A.apply_inverse_adjoint(y)
            return np.matrix(x.data).T

        def Es_apply(self, op, y):
            if self.E is None:
                return y

            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.E.apply_inverse(y)
            else:
                x = self.E.apply_inverse_adjoint(y)
            return np.matrix(x.data).T

        def ApE_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            if op == pymess.MESS_OP_NONE:
                x = self.A.apply(y)
                if self.E is None:
                    x += p * y
                else:
                    x += p * self.E.apply(y)
            else:
                x = self.A.apply_adjoint(y)
                if self.E is None:
                    x += p.conjugate() * y
                else:
                    x += p.conjugate() * self.E.apply_adjoint(y)
            return np.matrix(x.data).T

        def ApEs_apply(self, op, p, idx_p, y):
            y = self.A.source.from_data(np.array(y).T)
            E = IdentityOperator(self.A.source) if self.E is None else self.E

            if p.imag == 0:
                ApE = LincombOperator((self.A, E), (1, p.real))
            else:
                ApE = LincombOperator((self.A, E), (1, p))

            if op == pymess.MESS_OP_NONE:
                x = ApE.apply_inverse(y)
            else:
                x = ApE.apply_inverse_adjoint(y)
            return np.matrix(x.data).T

        def parameter(self, arp_p, arp_m, B=None, K=None):
            n = self.A.source.dim

            def A_mv(v):
                v = self.A.source.from_data(v)
                w = self.A.apply(v)
                return w.data[0, :]

            def A_rmv(v):
                v = self.A.range.from_data(v)
                w = self.A.apply_adjoint(v)
                return w.data[0, :]

            def A_mm(v):
                v = self.A.source.from_data(v.T)
                w = self.A.apply(v)
                return w.data.T

            A_scipy = spsla.LinearOperator((n, n), matvec=A_mv, rmatvec=A_rmv, matmat=A_mm, dtype=float)

            if self.E is None:
                E_scipy = None
            else:
                def E_mv(v):
                    v = self.E.source.from_data(v)
                    w = self.E.apply(v)
                    return w.data[0, :]

                def E_rmv(v):
                    v = self.E.range.from_data(v)
                    w = self.E.apply_adjoint(v)
                    return w.data[0, :]

                def E_mm(v):
                    v = self.E.source.from_data(v.T)
                    w = self.E.apply(v)
                    return w.data.T

                E_scipy = spsla.LinearOperator((n, n), matvec=E_mv, rmatvec=E_rmv, matmat=E_mm, dtype=float)

            lm = spsla.eigs(A_scipy, 20, E_scipy, which='LM', return_eigenvectors=False)
            sm = spsla.eigs(A_scipy, 20, E_scipy, which='SM', return_eigenvectors=False)

            # concatenate both and take real part, and filter positive ones
            ev = np.concatenate((lm, sm))
            ev = ev.real
            ev = ev[ev < 0]
            return ev
except ImportError:
    pass


def solve_lyap(A, E, B, trans=False, me_solver=None, tol=None):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor :math:`Z` such that :math:`Z Z^T` is approximately
    the solution :math:`X` of a Lyapunov equation (if E is `None`)

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
        Method to use ('scipy', 'slycot', 'pymess_lyap', 'pymess_lradi').

        If `me_solver` is `None`, a method is chosen according to availability and priority:

            'pymess_lradi' (for bigger problems) > 'pymess_lyap' (for smaller problems) > 'slycot' > 'scipy'.
    tol
        Tolerance parameter.

    Returns
    -------
    Z
        Low-rank factor of the Lyapunov equation solution, |VectorArray| from `A.source`.
    """
    assert isinstance(A, OperatorInterface) and A.linear
    assert A.source == A.range
    assert isinstance(B, OperatorInterface) and B.linear
    assert not trans and B.range == A.source or trans and B.source == A.source
    assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
    assert me_solver is None or me_solver in ('scipy', 'slycot', 'pymess_lyap', 'pymess_lradi')

    if me_solver is None:
        import imp
        try:
            imp.find_module('pymess')
            if A.source.dim >= 1000 or not isinstance(A, NumpyMatrixOperator):
                me_solver = 'pymess_lradi'
            else:
                me_solver = 'pymess_lyap'
        except ImportError:
            try:
                imp.find_module('slycot')
                me_solver = 'slycot'
            except ImportError:
                me_solver = 'scipy'

    if me_solver == 'scipy':
        if E is not None:
            raise NotImplementedError()
        import scipy.linalg as spla
        A_mat = to_matrix(A)
        B_mat = to_matrix(B)
        if not trans:
            X = spla.solve_lyapunov(A_mat, -B_mat.dot(B_mat.T))
        else:
            X = spla.solve_lyapunov(A_mat.T, -B_mat.T.dot(B_mat))
        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif me_solver == 'slycot':
        import slycot
        A_mat = to_matrix(A)
        if E is not None:
            E_mat = to_matrix(E)
        B_mat = to_matrix(B)

        n = A_mat.shape[0]
        if not trans:
            C = -B_mat.dot(B_mat.T)
            trana = 'T'
        else:
            C = -B_mat.T.dot(B_mat)
            trana = 'N'
        dico = 'C'

        if E is None:
            U = np.zeros((n, n))
            X, scale, _, _, _ = slycot.sb03md(n, C, A_mat, U, dico, trana=trana)
        else:
            job = 'X'
            fact = 'N'
            Q = np.zeros((n, n))
            Z = np.zeros((n, n))
            uplo = 'L'
            X = C
            _, _, _, _, X, scale, _, _, _, _, _ = slycot.sg03ad(dico, job, fact, trana, uplo, n, A_mat, E_mat,
                                                                Q, Z, X)

        from pymor.algorithms.cholp import cholp
        Z = cholp(X, copy=False)
    elif me_solver == 'pymess_lyap':
        import pymess
        A_mat = to_matrix(A)
        if E is not None:
            E_mat = to_matrix(E)
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
        import pymess
        opts = pymess.options()
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
