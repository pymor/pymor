# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interfaces import VectorArrayInterface
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
        The |Operator| A.
    B
        The |VectorArray| B.
    C
        The |VectorArray| C^T.
    D
        The |Operator| D or None (then D is assumed to be zero).
    E
        The |Operator| E or None (then E is assumed to be the identity).
    cont_time
        `True` if the system is continuous-time, otherwise discrete-time.

    Attributes
    ----------
    n
        Order of the system.
    m
        Number of inputs.
    p
        Number of outputs.
    """
    linear = True

    def __init__(self, A, B, C, D=None, E=None, cont_time=True):
        assert isinstance(A, OperatorInterface) and A.linear
        assert isinstance(B, VectorArrayInterface)
        assert isinstance(C, VectorArrayInterface)
        assert A.source == A.range and B in A.source and C in A.range
        assert (D is None or isinstance(D, OperatorInterface) and D.linear and D.source.dim == len(B) and
                D.range.dim == len(C))
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in {True, False}

        self.n = A.source.dim
        self.m = len(B)
        self.p = len(C)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.cont_time = cont_time
        self._w = None
        self._tfw = None
        self._cgf = None
        self._ogf = None
        self._hsv = None
        self._U = None
        self._V = None
        self.build_parameter_type(inherits=(A, D, E))

    @classmethod
    def from_matrices(cls, A, B, C, D=None, E=None, cont_time=True):
        """Creates LTISystem from matrices

        Parameters
        ----------
        A
            The |NumPy array|, |SciPy spmatrix| A.
        B
            The |NumPy array|, |SciPy spmatrix| B.
        C
            The |NumPy array|, |SciPy spmatrix| C.
        D
            The |NumPy array|, |SciPy spmatrix| D or None (then D is assumed to be zero).
        E
            The |NumPy array|, |SciPy spmatrix| E or None (then E is assumed to be the identity).
        cont_time
            `True` if the system is continuous-time, otherwise discrete-time.

        Returns
        -------
        lti
            LTISystem with operators A, B, C, D, and E.
        """
        assert isinstance(A, (np.ndarray, sps.spmatrix))
        assert isinstance(B, (np.ndarray, sps.spmatrix))
        assert isinstance(C, (np.ndarray, sps.spmatrix))
        assert D is None or isinstance(D, (np.ndarray, sps.spmatrix))
        assert E is None or isinstance(E, (np.ndarray, sps.spmatrix))

        A = NumpyMatrixOperator(A)
        B = NumpyVectorArray(B.T)
        C = NumpyVectorArray(C)
        if D is not None:
            D = NumpyMatrixOperator(D)
        if E is not None:
            E = NumpyMatrixOperator(E)

        return cls(A, B, C, D, E, cont_time)

    @classmethod
    def from_mat_file(cls, file_name, cont_time=True):
        """Creates LTISystem from matrices

        Parameters
        ----------
        file_name
            Name of the mat file (extension .mat does not need to be included)
            containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise discrete-time.

        Returns
        -------
        lti
            LTISystem with operators A, B, C, D, and E.
        """
        import scipy.io as spio
        mat_dict = spio.loadmat(file_name)

        assert 'A' in mat_dict and 'B' in mat_dict and 'C' in mat_dict

        A = mat_dict['A']
        B = mat_dict['B']
        C = mat_dict['C']
        if 'D' in mat_dict:
            D = mat_dict['D']
        else:
            D = None
        if 'E' in mat_dict:
            E = mat_dict['E']
        else:
            E = None

        return cls.from_matrices(A, B, C, D, E, cont_time)

    def _solve(self, mu=None):
        raise NotImplementedError('Discretization has no solver.')

    def __add__(self, other):
        """Add two LTI systems"""
        assert isinstance(other, LTISystem)
        assert self.m == other.m and self.p == other.p
        assert self.cont_time == other.cont_time

        # form A
        if not self.A.sparse and not other.A.sparse:
            A = spla.block_diag(self.A._matrix, other.A._matrix)
        else:
            A = sps.block_diag((self.A._matrix, other.A._matrix))
            A = A.tocsc()

        # form B
        if not sps.issparse(self.B.data) and not sps.issparse(other.B.data):
            B = sp.vstack((self.B.data.T, other.B.data.T))
        else:
            B = sps.vstack((sps.coo_matrix(self.B.data.T), sps.coo_matrix(other.B.data.T)))
            B = B.tocsc()

        # form C
        if not sps.issparse(self.C.data) and not sps.issparse(other.C.data):
            C = sp.hstack((self.C.data, other.C.data))
        else:
            C = sps.hstack((sps.coo_matrix(self.C.data), sps.coo_matrix(other.C.data)))
            C = C.tocsc()

        # form D
        if self.D is None and other.D is None:
            D = None
        elif self.D is not None and other.D is None:
            D = self.D
        elif self.D is None and other.D is not None:
            D = other.D
        else:
            D = self.D + other.D

        # form E
        if self.E is None and other.E is None:
            E = None
        elif self.E is None and other.E is not None:
            E = sps.block_diag((sps.eye(self.n), other.E._matrix))
            E = E.tocsc()
        elif self.E is not None and other.E is None:
            E = sps.block_diag((self.E._matrix, sps.eye(other.n)))
            E = E.tocsc()
        elif not self.E.sparse and not other.E.sparse:
            E = spla.block_diag(self.E._matrix, other.E._matrix)
        else:
            E = sps.block_diag((self.E._matrix, other.E._matrix))
            E = E.tocsc()

        return LTISystem.from_matrices(A, B, C, D, E, self.cont_time)

    def __neg__(self):
        """Negate LTI system"""
        A = self.A
        B = self.B
        C = NumpyVectorArray(-self.C.data)
        D = self.D
        if D is not None:
            D = NumpyMatrixOperator(-D._matrix)
        E = self.E

        return LTISystem(A, B, C, D, E, self.cont_time)

    def __sub__(self, other):
        """Subtract two LTI systems"""
        return self + (-other)

    def bode(self, w):
        """Computes the Bode plot

        Parameters
        ----------
        w
            Frequencies at which to compute the transfer function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in w, returned as a 3D |NumPy array| of shape (p, m, len(w)).
        """
        if not self.cont_time:
            raise NotImplementedError

        self._w = w
        self._tfw = np.zeros((self.p, self.m, len(w)), dtype=complex)

        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        if E is None:
            if not A.sparse:
                import scipy as sp
                E = NumpyMatrixOperator(sp.eye(self.n))
            else:
                import scipy.sparse as sps
                E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))

        for i in xrange(len(w)):
            iwEmA = A.assemble_lincomb((E, A), (1j * w[i], -1))
            G = C.dot(iwEmA.apply_inverse(B))
            if D is not None:
                G += D._matrix
            self._tfw[:, :, i] = G

        return self._tfw.copy()

    def compute_cgf(self):
        """Computes the controllability gramian factor"""
        if not self.cont_time:
            raise NotImplementedError

        if self._cgf is None:
            self._cgf = solve_lyap(self.A, self.E, self.B)

    def compute_ogf(self):
        """Computes the observability gramian factor"""
        if not self.cont_time:
            raise NotImplementedError

        if self._ogf is None:
            self._ogf = solve_lyap(self.A, self.E, self.C, trans=True)

    def compute_hsv_U_V(self):
        """Compute Hankel singular values and vectors"""
        if self._hsv is None or self._U is None or self._V is None:
            self.compute_cgf()
            self.compute_ogf()

            if self.E is None:
                self._U, self._hsv, Vh = spla.svd(self._cgf.dot(self._ogf))
            else:
                self._U, self._hsv, Vh = spla.svd(self.E.apply2(self._cgf, self._ogf))
            self._V = Vh.T

    def norm(self, name='H2'):
        """Computes a norm of the LTI system

        Parameters
        ----------
        name
            Name of the norm ('H2')
        """
        if name == 'H2':
            import numpy.linalg as npla

            if self._cgf is not None:
                return npla.norm(self.C.dot(self._cgf))
            if self._ogf is not None:
                return npla.norm(self.B.dot(self._ogf))
            if self.m <= self.p:
                self.compute_cgf()
                return npla.norm(self.C.dot(self._cgf))
            else:
                self.compute_ogf()
                return npla.norm(self.B.dot(self._ogf))
        else:
            raise NotImplementedError('Only H2 norm is implemented.')


class LyapunovEquation(pymess.equation):
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
        The |VectorArray| RHS.
    """
    def __init__(self, opt, A, E, RHS):
        dim = A.source.dim

        super(LyapunovEquation, self).__init__(name='lyap_eqn', opt=opt, dim=dim)

        self.A = A
        self.E = E
        self.RHS = np.array(RHS.data.T)
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


def solve_lyap(A, E, B, trans=False):
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
        The |VectorArray| B.
    """
    opts = pymess.options()
    if trans:
        opts.type = pymess.MESS_OP_TRANSPOSE
    eqn = LyapunovEquation(opts, A, E, B)
    Z, status = pymess.lradi(eqn, opts)
    Z = NumpyVectorArray(np.array(Z).T)

    return Z.copy()
