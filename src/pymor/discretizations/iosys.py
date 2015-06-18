# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse as sps

from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorArray

import pymess


class LTISystem(DiscretizationInterface):
    """Class for linear time-invariant systems

    This class describes input-state-output systems given by::

        E x'(t) = A x(t) + B u(t)
           y(t) = C x(t) + D u(t)

    if continuous-time, or::

        E x(k + 1) = A x(k) + B u(k)
          y(k)     = C x(k) + D u(k)

    if discrete-time, where A, B, C, D, and E are linear operators.

    Parameters
    ----------
    A
        The |NumPy array|, |SciPy spmatrix| or |Operator| A.
    B
        The |NumPy array|, |SciPy spmatrix| or |Operator| B.
    C
        The |NumPy array|, |SciPy spmatrix| or |Operator| C.
    D
        The |NumPy array|, |SciPy spmatrix| or |Operator| D or None (then D is assumed to be zero).
    E
        The |NumPy array|, |SciPy spmatrix| or |Operator| E or None (then E is assumed to be the identity).
    cont_time
        `True` if the system is continuous-time, otherwise discrete-time.

    Attributes
    ----------
    n
        Size of x.
    """
    linear = True

    def __init__(self, A, B, C, D=None, E=None, cont_time=True):
        if isinstance(A, (np.ndarray, sps.spmatrix)):
            A = NumpyMatrixOperator(A)
        if isinstance(B, (np.ndarray, sps.spmatrix)):
            B = NumpyMatrixOperator(B)
        if isinstance(C, (np.ndarray, sps.spmatrix)):
            C = NumpyMatrixOperator(C)
        if isinstance(D, (np.ndarray, sps.spmatrix)):
            D = NumpyMatrixOperator(D)
        if isinstance(E, (np.ndarray, sps.spmatrix)):
            E = NumpyMatrixOperator(E)

        assert isinstance(A, OperatorInterface) and A.linear
        assert isinstance(B, OperatorInterface) and B.linear
        assert isinstance(C, OperatorInterface) and C.linear
        assert A.source == A.range == B.range == C.source
        assert (D is None or isinstance(D, OperatorInterface) and D.linear and D.source == B.source and
                D.range == C.range)
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in {True, False}

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.cont_time = cont_time
        self.build_parameter_type(inherits=(A, B, C, D, E))

    def _solve(self):
        raise NotImplementedError('Discretization has no solver.')

    def norm(self):
        if self.cont_time:
            import numpy.linalg as npla
            Z = solve_lyap(self.A, self.E, self.B)
            return npla.norm(self.C.apply(Z).l2_norm())
        else:
            raise NotImplementedError


class eqn_lyap(pymess.equation):
    """Lyapunov equation class for pymess

    Represents a Lyapunov equation::

        A * X + X * A^T + RHS * RHS^T = 0

    if E is None, otherwise a generalized Lyapunov equation::

        A * X * E^T + E * X * A^T + RHS * RHS^T = 0.

    For the dual Lyapunov equation, opt.type needs to be pymess.MESS_OP_TRANSPOSE.

    Parameters
    ----------
    opt
        pymess options structure
    A
        The |Operator| A.
    E
        The |Operator| E or None.
    RHS
        The |Operator| RHS.
    """
    def __init__(self, opt, A, E, RHS):
        dim = A.source.dim

        super(eqn_lyap, self).__init__(name="eqn_lyap", opt=opt, dim=dim)

        self.A = A
        self.E = E
        self.RHS = np.array(RHS._matrix)
        self.p = []

    def A_apply(self, op, y):
        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply(y)
        else:
            x = self.A.apply_adjoint(y)
        return np.matrix(x.data).T

    def E_apply(self, op, y):
        if self.E is None:
            return y

        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply(y)
        else:
            x = self.E.apply_adjoint(y)
        return np.matrix(x.data).T

    def As_apply(self, op, y):
        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply_inverse(y)
        else:
            x = self.A.apply_adjoint_inverse(y)
        return np.matrix(x.data).T

    def Es_apply(self, op, y):
        if self.E is None:
            return y

        y = NumpyVectorArray(np.array(y).T)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply_inverse(y)
        else:
            x = self.E.apply_adjoint_inverse(y)
        return np.matrix(x.data).T

    def ApE_apply(self, op, p, idx_p, y):
        y = NumpyVectorArray(np.array(y).T)
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
        y = NumpyVectorArray(np.array(y).T)
        if self.E is None:
            E = NumpyMatrixOperator(sps.eye(self.dim))
        else:
            E = self.E

        if p.imag == 0.0:
            ApE = self.A.assemble_lincomb((self.A, E), (1, p.real))
        else:
            ApE = self.A.assemble_lincomb((self.A, E), (1, p))

        if op == pymess.MESS_OP_NONE:
            x = ApE.apply_inverse(y)
        else:
            x = ApE.apply_adjoint_inverse(y)
        return np.matrix(x.data).T


def solve_lyap(A, E, B, trans=True):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a Lyapunov equation::

        A * X + X * A^T + B * B^T = 0

    if E is None, otherwise a generalized Lyapunov equation::

        A * X * E^T + E * X * A^T + B * B^T = 0.

    If trans is True, then solve::

        A^T * X + X * A + B^T * B = 0

    if E is None, otherwise::

        A^T * X * E + E^T * X * A + B^T * B = 0.

    Parameters
    ----------
    A
        The |Operator| A.
    E
        The |Operator| E or None.
    B
        The |Operator| B.
    """
    opts = pymess.options()
    if trans:
        opts.type = pymess.MESS_OP_TRANSPOSE
    eqn = eqn_lyap(opts, A, E, B)
    Z, status = pymess.lradi(eqn, opts)
    Z = NumpyVectorArray(np.array(Z).T)

    return Z.copy()
