# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Rene Milk, Stephan Rave, Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.lyapunov import solve_lyap
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.constructions import VectorArrayOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import GenericRBReconstructor
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorArray


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
        The |Operator| D or `None` (then D is assumed to be zero).
    E
        The |Operator| E or `None` (then E is assumed to be the identity).
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
        """Create |LTISystem| from matrices.

        Parameters
        ----------
        A
            The |NumPy array|, |SciPy spmatrix| A.
        B
            The |NumPy array|, |SciPy spmatrix| B.
        C
            The |NumPy array|, |SciPy spmatrix| C.
        D
            The |NumPy array|, |SciPy spmatrix| D or `None` (then D is assumed to be zero).
        E
            The |NumPy array|, |SciPy spmatrix| E or `None` (then E is assumed to be the identity).
        cont_time
            `True` if the system is continuous-time, otherwise discrete-time.

        Returns
        -------
        lti
            |LTISystem| with operators A, B, C, D, and E.
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
        """Create |LTISystem| from matrices stored in a .mat file.

        Parameters
        ----------
        file_name
            Name of the .mat file (extension .mat does not need to be included)
            containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise discrete-time.

        Returns
        -------
        lti
            |LTISystem| with operators A, B, C, D, and E.
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
        """Add two |LTISystems|."""
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
        """Negate |LTISystem|."""
        A = self.A
        B = self.B
        C = NumpyVectorArray(-self.C.data)
        D = self.D
        if D is not None:
            D = NumpyMatrixOperator(-D._matrix)
        E = self.E

        return LTISystem(A, B, C, D, E, self.cont_time)

    def __sub__(self, other):
        """Subtract two |LTISystems|."""
        return self + (-other)

    def bode(self, w):
        """Compute the transfer function on the imaginary axis.

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
        """Compute the controllability gramian factor."""
        if not self.cont_time:
            raise NotImplementedError

        if self._cgf is None:
            self._cgf = solve_lyap(self.A, self.E, self.B)

    def compute_ogf(self):
        """Compute the observability gramian factor."""
        if not self.cont_time:
            raise NotImplementedError

        if self._ogf is None:
            self._ogf = solve_lyap(self.A, self.E, self.C, trans=True)

    def compute_hsv_U_V(self):
        """Compute the Hankel singular values and vectors."""
        if self._hsv is None or self._U is None or self._V is None:
            self.compute_cgf()
            self.compute_ogf()

            if self.E is None:
                U, self._hsv, Vh = spla.svd(self._cgf.dot(self._ogf))
            else:
                U, self._hsv, Vh = spla.svd(self.E.apply2(self._cgf, self._ogf))

            self._U = NumpyVectorArray(U.T)
            self._V = NumpyVectorArray(Vh)

    def norm(self, name='H2'):
        """Compute a norm of the |LTISystem|.

        Parameters
        ----------
        name
            Name of the norm ('H2')
        """
        if name == 'H2':
            if self._cgf is not None:
                return spla.norm(self.C.dot(self._cgf))
            if self._ogf is not None:
                return spla.norm(self.B.dot(self._ogf))
            if self.m <= self.p:
                self.compute_cgf()
                return spla.norm(self.C.dot(self._cgf))
            else:
                self.compute_ogf()
                return spla.norm(self.B.dot(self._ogf))
        else:
            raise NotImplementedError('Only H2 norm is implemented.')

    def project(self, Vr, Wr):
        """Reduce using Petrov-Galerkin projection.

        Parameters
        ----------
        Vr
            Right projection matrix.
        Wr
            Left projection matrix.

        Returns
        -------
        Ar
            |NumPy array| of size r x r.
        Br
            |NumPy array| of size r x m.
        Cr
            |NumPy array| of size p x r.
        Dr
            |NumPy array| of size p x m or None.
        Er
            |NumPy array| of size r x r.
        """
        Ar = self.A.apply2(Wr, Vr)
        Br = Wr.dot(self.B)
        Cr = self.C.dot(Vr)
        if self.D is None:
            Dr = None
        else:
            Dr = self.D.copy()
        if self.E is None:
            Er = Wr.dot(Vr)
        else:
            Er = self.E.apply2(Wr, Vr)

        return Ar, Br, Cr, Dr, Er

    def bt(self, r):
        """Reduce using the balanced truncation method to order r.

        Parameters
        ----------
        r
            Order of the reduced model.

        Returns
        -------
        rom
            Reduced |LTISystem|.
        rc
            The reconstructor providing a `reconstruct(U)` method which reconstructs
            high-dimensional solutions from solutions `U` of the reduced |LTISystem|.
        reduction_data
            Additional data produced by the reduction process. Contains projection matrices `Vr` and `Wr`.
        """
        assert 0 < r < self.n

        self.compute_hsv_U_V()

        Vr = VectorArrayOperator(self._cgf).apply(self._U, ind=range(r))
        Wr = VectorArrayOperator(self._ogf).apply(self._V, ind=range(r))
        Vr = gram_schmidt(Vr, atol=0, rtol=0)
        Wr = gram_schmidt(Wr, atol=0, rtol=0)

        Ar, Br, Cr, Dr, Er = self.project(Vr, Wr)

        rom = LTISystem.from_matrices(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr}

        return rom, rc, reduction_data

    def interpolation(self, sigma, b, c):
        """Find `Vr` and `Wr`.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of length `r`.
        b
            Right tangential directions, |NumPy array| of order `m x r`.
        c
            Left tangential directions, |NumPy array| of order `p x r`.

        Returns
        -------
        Vr
            Right projection matrix.
        Wr
            Left projection matrix.
        """
        r = len(sigma)

        Vr = NumpyVectorArray.make_array(self.n, reserve=r)
        Wr = NumpyVectorArray.make_array(self.n, reserve=r)

        for i in xrange(r):
            if sigma[i].imag == 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i].real, -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i].real, -1))

                Bb = VectorArrayOperator(self.B).apply(NumpyVectorArray(b[:, i].real.T))
                Vr.append(sEmA.apply_inverse(Bb))
                CTc = VectorArrayOperator(self.C).apply(NumpyVectorArray(c[:, i].real.T))
                Wr.append(sEmA.apply_adjoint_inverse(CTc))
            elif sigma[i].imag > 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i], -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i], -1))

                Bb = VectorArrayOperator(self.B).apply(NumpyVectorArray(b[:, i].T))
                v = sEmA.apply_inverse(Bb)
                Vr.append(NumpyVectorArray(v.data.real))
                Vr.append(NumpyVectorArray(v.data.imag))

                CTc = VectorArrayOperator(self.C).apply(NumpyVectorArray(c[:, i].T))
                w = sEmA.apply_adjoint_inverse(CTc)
                Wr.append(NumpyVectorArray(w.data.real))
                Wr.append(NumpyVectorArray(w.data.imag))

        Vr = gram_schmidt(Vr, atol=0, rtol=0)
        Wr = gram_schmidt(Wr, atol=0, rtol=0)

        return Vr, Wr

    def irka(self, sigma, b, c, tol, maxit, prnt=False):
        """Reduce using IRKA.

        Parameters
        ----------
        sigma
            Initial interpolation points (closed under conjugation), list of length `r`.
        b
            Initial right tangential directions, |NumPy array| of order `m x r`.
        c
            Initial left tangential directions, |NumPy array| of order `p x r`.
        tol
            Tolerance, largest change in interpolation points.
        maxit
            Maximum number of iterations.
        prnt
            Should consecutive distances be printed.

        Returns
        -------
        rom
            Reduced |LTISystem| model.
        rc
            Reconstructor of full state.
        reduction_data
            Dictionary of additional data produced by the reduction process. Contains
            projection matrices `Vr` and `Wr`, distances between interpolation points in
            different iterations `dist`, and interpolation points from all iterations `Sigma`.
        """
        Vr, Wr = self.interpolation(sigma, b, c)

        dist = []
        Sigma = [np.array(sigma)]
        for it in xrange(maxit):
            Ar, Br, Cr, _, Er = self.project(Vr, Wr)

            sigma, Y, X = spla.eig(Ar, Er, left=True, right=True)
            sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            Sigma.append(sigma.copy())

            dist.append([])
            for i in xrange(it + 1):
                dist[-1].append(np.max(np.abs((Sigma[i] - Sigma[-1]) / Sigma[-1])))

            if prnt:
                print('dist[{}] = {:.5e}'.format(it, np.min(dist[-1])))

            b = Br.T.dot(Y.conj())
            c = Cr.dot(X)

            Vr, Wr = self.interpolation(sigma, b, c)

            if np.min(dist[-1]) < tol:
                break

        Ar, Br, Cr, Dr, Er = self.project(Vr, Wr)

        rom = LTISystem.from_matrices(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr, 'dist': dist, 'Sigma': Sigma, 'b': b, 'c': c}

        return rom, rc, reduction_data
