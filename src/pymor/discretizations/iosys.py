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


class IOSystem(DiscretizationInterface):
    """Base class for input-output systems

    Attributes
    ----------
    cont_time
        `True` if the system is continuous-time, otherwise discrete-time.
    m
        Number of inputs.
    p
        Number of outputs.
    """
    cont_time = None
    m = None
    p = None


class LTISystem(IOSystem):
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
    """docstring for my_equation"""
    def __init__(self, opt, A, E, RHS):
        dim = A.source.dim

        super(eqn_lyap, self).__init__(name="eqn_lyap", opt=opt, dim=dim)

        self.RHS = RHS
        self.A = A
        self.E = E
        self.p = []

    def A_apply(self, op, y):
        """docstring for A_apply"""
        print("A_apply")

        y = NumpyVectorArray(y)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply(y)
        else:
            x = self.A.apply_adjoint(y)
        return x.data

    def E_apply(self, op, y):
        print("E_apply")

        if self.E is None:
            return y

        y = NumpyVectorArray(y)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply(y)
        else:
            x = self.E.apply_adjoint(y)
        return x.data

    def As_apply(self, op, y):
        print("As_apply")

        y = NumpyVectorArray(y)
        if op == pymess.MESS_OP_NONE:
            x = self.A.apply_inverse(y)
        else:
            x = self.A.apply_adjoint_inverse(y)
        return x.data

    def Es_apply(self, op, y):
        print("Es_apply")

        if self.E is None:
            return y

        y = NumpyVectorArray(y)
        if op == pymess.MESS_OP_NONE:
            x = self.E.apply_inverse(y)
        else:
            x = self.E.apply_adjoint_inverse(y)
        return x.data

    def ApE_apply(self, op, p, idx_p, y):
        print("Apply ApE p = %f (idx = %d)" % (p, idx_p))

        y = NumpyVectorArray(y)
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
        return x.data

    def ApEs_apply(self, op, p, idx_p, y):
        print("ApEs_apply - op = %s" % (op))
        if self.E is None:
            E = NumpyMatrixOperator(sps.eye(self.A.source.dim))
        else:
            E = self.E

        if p.imag == 0.0:
            ApE = self.A.assemble_lincomb((self.A, E), (1, p.real))
        else:
            ApE = self.A.assemble_lincomb((self.A, E), (1, p))

        if op == pymess.MESS_OP_NONE:
            return ApE.apply_inverse(y)
        else:
            return ApE.apply_adjoint_inverse(y)


def solve_lyap(A, E, B):
    """Find a factor of the solution of a Lyapunov equation

    Returns factor Z such that Z * Z^T is approximately the solution X of a Lyapunov equation::

        A * X + X * A^T + B * B^T = 0

    if E is None, otherwise a generalized Lyapunov equation::

        A * X * E^T + E * X * A^T + B * B^T = 0.

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
    opts.type = pymess.MESS_OP_NONE
    eqn = eqn_lyap(opts, A, E, B)
    Z, status = pymess.lradi(eqn, opts)
    Z = np.array(Z)
    Z = NumpyVectorArray(Z)

    return Z
