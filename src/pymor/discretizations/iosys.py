# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
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
        The |Operator| B.
    C
        The |Operator| C.
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
        assert A.source == A.range
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == A.source
        assert isinstance(C, OperatorInterface) and C.linear
        assert C.source == A.range
        assert (D is None or
                isinstance(D, OperatorInterface) and D.linear and D.source == B.source and D.range == C.range)
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
        self._w = None
        self._tfw = None
        self._cgf = None
        self._ogf = None
        self._hsv = None
        self._U = None
        self._V = None
        self._brgamma = 1.
        self._brcgf = None
        self._brogf = None
        self._brsv = None
        self._brU = None
        self._brV = None
        self._H2_norm = None
        self._Hinf_norm = None
        self._fpeak = None
        self.build_parameter_type(inherits=(A, B, C, D, E))

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
        B = NumpyMatrixOperator(B)
        C = NumpyMatrixOperator(C)
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
        if not self.B.sparse and not other.B.sparse:
            B = sp.vstack((self.B._matrix, other.B._matrix))
        else:
            B = sps.vstack((sps.coo_matrix(self.B._matrix), sps.coo_matrix(other.B._matrix)))
            B = B.tocsc()

        # form C
        if not self.C.sparse and not other.C.sparse:
            C = sp.hstack((self.C._matrix, other.C._matrix))
        else:
            C = sps.hstack((sps.coo_matrix(self.C._matrix), sps.coo_matrix(other.C._matrix)))
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
        C = NumpyMatrixOperator(-self.C._matrix)
        D = self.D
        if D is not None:
            D = NumpyMatrixOperator(-D._matrix)
        E = self.E

        return LTISystem(A, B, C, D, E, self.cont_time)

    def __sub__(self, other):
        """Subtract two |LTISystems|."""
        return self + (-other)

    def eval_tf(self, s):
        """Evaluate the transfer function

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaulated at the complex number s, 2D |NumPy array|.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        if E is None:
            if not A.sparse:
                E = NumpyMatrixOperator(sp.eye(self.n))
            else:
                E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))

        iwEmA = A.assemble_lincomb((E, A), (s, -1))
        if self.m <= self.p:
            eye = NumpyVectorArray(sp.eye(self.m))
            tfs = C.apply(iwEmA.apply_inverse(B.apply(eye))).data.T
        else:
            eye = NumpyVectorArray(sp.eye(self.p))
            tfs = B.apply_adjoint(iwEmA.apply_adjoint_inverse(C.apply_adjoint(eye))).data
        if D is not None:
            tfs += D._matrix
        return tfs

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
        self._tfw = np.empty((self.p, self.m, len(w)), dtype=complex)

        for i, wi in enumerate(w):
            self._tfw[:, :, i] = self.eval_tf(1j * wi)

        return self._tfw.copy()

    @classmethod
    def mag_plot(cls, sys_list, plot_style_list=None, w=None, ord=None, dB=False, Hz=False):
        """Draw the magnitude Bode plot

        Parameters
        ----------
        sys_list
            A single |LTISystem| or a list of |LTISystems|.
        plot_style_list
            A string or a list of strings of the same length as `sys_list`.
            If None, matplotlib defaults are used.
        w
            Frequencies at which to compute the transfer function.
            If None, use self._w.
        ord
            Order of the norm used to compute the magnitude (the default is the Frobenius norm).
        dB
            Should the magnitude be in dB on the plot.
        Hz
            Should the frequency be in Hz on the plot.
        """
        assert isinstance(sys_list, LTISystem) or all(isinstance(sys, LTISystem) for sys in sys_list)
        if isinstance(sys_list, LTISystem):
            sys_list = (sys_list,)

        assert (plot_style_list is None or isinstance(plot_style_list, basestring) or
                all(isinstance(plot_style, basestring) for plot_style in plot_style_list))
        if isinstance(plot_style_list, basestring):
            plot_style_list = (plot_style_list,)

        assert w is not None or all(sys._w is not None for sys in sys_list)

        if w is not None:
            for sys in sys_list:
                sys.bode(w)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, sys in enumerate(sys_list):
            freq = sys._w
            if Hz:
                freq = freq.copy() / (2 * np.pi)
            mag = np.array([spla.norm(sys._tfw[:, :, j], ord=ord) for j, _ in enumerate(freq)])
            style = '' if plot_style_list is None else plot_style_list[i]
            if dB:
                mag = 20 * np.log2(mag)
                ax.semilogx(freq, mag, style)
            else:
                ax.loglog(freq, mag, style)
        ax.set_title('Magnitude Bode Plot')
        if Hz:
            ax.set_xlabel('Frequency (Hz)')
        else:
            ax.set_xlabel('Frequency (rad/s)')
        if dB:
            ax.set_ylabel('Magnitude (dB)')
        else:
            ax.set_ylabel('Magnitude')
        plt.show()
        return fig, ax

    def compute_cgf(self, tol=None):
        """Compute the controllability Gramian factor.

        Parameters
        ----------
        tol
            Tolerance parameter for the low-rank Lyapunov equation solver.
            If None, then the default tolerance is used. Otherwise, it should be a positive float and
            the controllability Gramian factor is recomputed (if it was already computed).
        """
        if not self.cont_time:
            raise NotImplementedError

        if self._cgf is None or tol is not None:
            self._cgf = solve_lyap(self.A, self.E, self.B, tol=tol)

    def compute_ogf(self, tol=None):
        """Compute the observability Gramian factor.

        Parameters
        ----------
        tol
            Tolerance parameter for the low-rank Lyapunov equation solver.
            If None, then the default tolerance is used. Otherwise, it should be a positive float and
            the observability Gramian factor is recomputed (if it was already computed).
        """
        if not self.cont_time:
            raise NotImplementedError

        if self._ogf is None or tol is not None:
            self._ogf = solve_lyap(self.A, self.E, self.C, trans=True, tol=tol)

    def compute_hsv_U_V(self):
        """Compute the Hankel singular values and vectors."""
        if self._hsv is None or self._U is None or self._V is None:
            self.compute_cgf()
            self.compute_ogf()

            if self.E is None:
                U, self._hsv, Vh = spla.svd(self._ogf.dot(self._cgf))
            else:
                U, self._hsv, Vh = spla.svd(self.E.apply2(self._ogf, self._cgf))

            self._U = NumpyVectorArray(U.T)
            self._V = NumpyVectorArray(Vh)

    def compute_brcgf(self, gamma=1., tol=1e-14):
        """Compute the bounded real controllability Gramian factor.

        Parameters
        ----------
        gamma
            Bound for the H_inf norm.
        tol
            Tolerance for the low-rank approximation.
        """
        assert gamma > 0 and 0 < tol < 1

        if not self.cont_time or self.E is not None:
            raise NotImplementedError

        if self._brcgf is None or self._brgamma != gamma:
            self._brgamma = gamma
            A = self.A._matrix
            if sps.issparse(A):
                A = A.toarray()
            B = self.B._matrix
            if sps.issparse(B):
                B = B.toarray()
            C = self.C._matrix
            if sps.issparse(C):
                C = C.toarray()
            X = spla.solve_continuous_are(A.T,
                                          C.T / np.sqrt(gamma),
                                          B.dot(B.T) / gamma,
                                          -np.eye(self.p))
            X = (X + X.T) / 2
            T, Z = spla.schur(X)
            ind = T.diagonal().argsort()[::-1]
            T = np.diag(np.diag(T)[ind])
            Z = Z[:, ind]
            rank = self.n
            for i in xrange(1, self.n):
                if T[i, i] / T[0, 0] < tol:
                    rank = i
                    break
            self._brcgf = NumpyVectorArray(Z[:, :rank].dot(np.diag(np.sqrt(T.diagonal()[:rank]))).T)

    def compute_brogf(self, gamma=1., tol=1e-14):
        """Compute the bounded real observability Gramian factor.

        Parameters
        ----------
        gamma
            Bound for the H_inf norm.
        tol
            Tolerance for the low-rank approximation.
        """
        assert gamma > 0 and 0 < tol < 1

        if not self.cont_time or self.E is not None:
            raise NotImplementedError

        if self._brogf is None or self._brgamma != gamma:
            self._brgamma = gamma
            A = self.A._matrix
            if sps.issparse(A):
                A = A.toarray()
            B = self.B._matrix
            if sps.issparse(B):
                B = B.toarray()
            C = self.C._matrix
            if sps.issparse(C):
                C = C.toarray()
            X = spla.solve_continuous_are(A,
                                          B / np.sqrt(gamma),
                                          C.T.dot(C) / gamma,
                                          -np.eye(self.m))
            X = (X + X.T) / 2
            T, Z = spla.schur(X)
            ind = T.diagonal().argsort()[::-1]
            T = np.diag(np.diag(T)[ind])
            Z = Z[:, ind]
            rank = self.n
            for i in xrange(1, self.n):
                if T[i, i] / T[0, 0] < tol:
                    rank = i
                    break
            self._brogf = NumpyVectorArray(Z[:, :rank].dot(np.diag(np.sqrt(T.diagonal()[:rank]))).T)

    def compute_brsv_brU_brV(self, gamma=1.):
        """Compute the bounded real singular values and vectors.

        Parameters
        ----------
        gamma
            Bound for the H_inf norm.
        """
        if self._brsv is None or self._brU is None or self._brV is None:
            self.compute_brcgf(gamma=gamma)
            self.compute_brogf(gamma=gamma)

            U, self._brsv, Vh = spla.svd(self._brogf.dot(self._brcgf))
            self._brU = NumpyVectorArray(U.T)
            self._brV = NumpyVectorArray(Vh)

    def norm(self, name='H2'):
        """Compute a norm of the |LTISystem|.

        Parameters
        ----------
        name
            Name of the norm ('H2', 'Hinf', 'Hankel')
        """
        if name == 'H2':
            if self._H2_norm is not None:
                return self._H2_norm

            if self._cgf is not None:
                self._H2_norm = spla.norm(self.C.apply(self._cgf).data)
            elif self._ogf is not None:
                self._H2_norm = spla.norm(self.B.apply_adjoint(self._ogf).data)
            elif self.m <= self.p:
                self.compute_cgf()
                self._H2_norm = spla.norm(self.C.apply(self._cgf).data)
            else:
                self.compute_ogf()
                self._H2_norm = spla.norm(self.B.apply_adjoint(self._ogf).data)
            return self._H2_norm
        elif name == 'Hinf':
            if self._Hinf_norm is not None:
                return self._Hinf_norm

            from slycot import ab13dd
            dico = 'C' if self.cont_time else 'D'
            jobe = 'I' if self.E is None else 'G'
            equil = 'S'
            jobd = 'Z' if self.D is None else 'D'
            A = self.A._matrix
            if self.A.sparse:
                A = A.toarray()
            B = self.B._matrix
            if self.B.sparse:
                B = B.toarray()
            C = self.C._matrix
            if self.C.sparse:
                C = C.toarray()
            if self.D is None:
                D = np.zeros((self.p, self.m))
            else:
                D = self.D._matrix
                if self.D.sparse:
                    D = D.toarray()
            if self.E is None:
                E = np.eye(self.n)
            else:
                E = self.E._matrix
                if self.E.sparse:
                    E = E.toarray()
            self._Hinf_norm, self._fpeak = ab13dd(dico, jobe, equil, jobd, self.n, self.m, self.p, A, E, B, C, D)
            return self._Hinf_norm
        elif name == 'Hankel':
            self.compute_hsv_U_V()
            return self._hsv[0]
        else:
            raise NotImplementedError('Only H2, Hinf, and Hankel norms are implemented.')

    def project(self, Vr, Wr, Er_is_identity=False):
        """Reduce using Petrov-Galerkin projection.

        Parameters
        ----------
        Vr
            Right projection matrix.
        Wr
            Left projection matrix.
        Er_is_identity
            If the reduced `E` is guaranteed to be the identity matrix.

        Returns
        -------
        Ar
            |NumPy array| of size `r x r`.
        Br
            |NumPy array| of size `r x m`.
        Cr
            |NumPy array| of size `p x r`.
        Dr
            |NumPy array| of size `p x m` or `None`.
        Er
            |NumPy array| of size `r x r`.
        """
        Ar = self.A.apply2(Wr, Vr)

        Br = self.B.apply_adjoint(Wr).data

        Cr = self.C.apply(Vr).data.T

        if self.D is None:
            Dr = None
        else:
            Dr = self.D.copy()

        if Er_is_identity:
            Er = None
        else:
            if self.E is None:
                Er = Wr.dot(Vr)
            else:
                Er = self.E.apply2(Wr, Vr)

        return Ar, Br, Cr, Dr, Er

    def bt(self, r=None, tol=None, meth='bfsr'):
        """Reduce using the Balanced Truncation method to order `r` or with tolerance `tol`.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is None.
        tol
            Tolerance for the absolute H_inf-error if `r` is None.
        meth
            Method used:
            * square root method ('sr')
            * balancing-free square root method ('bfsr')

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
        assert r is not None and tol is None or r is None and tol is not None
        assert r is None or 0 < r < self.n
        assert meth in {'sr', 'bfsr'}

        self.compute_cgf()
        self.compute_ogf()

        if r is not None and r > min([len(self._cgf), len(self._ogf)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.\
                              Try reducing the tolerance in the low-rank Lyapunov equation solver.')

        self.compute_hsv_U_V()

        if r is None:
            bounds = np.cumsum(self._hsv)
            bounds = bounds[-1] - bounds
            bounds *= 2
            r = min(i + 1 for i, b in enumerate(bounds) if b <= tol)

        Vr = VectorArrayOperator(self._cgf).apply(self._V, ind=range(r))
        Wr = VectorArrayOperator(self._ogf).apply(self._U, ind=range(r))

        if meth == 'sr':
            alpha = 1 / np.sqrt(self._hsv[:r])
            Vr.scal(alpha)
            Wr.scal(alpha)

            Ar, Br, Cr, Dr, Er = self.project(Vr, Wr, Er_is_identity=True)
        elif meth == 'bfsr':
            Vr = gram_schmidt(Vr, atol=0, rtol=0)
            Wr = gram_schmidt(Wr, atol=0, rtol=0)

            Ar, Br, Cr, Dr, Er = self.project(Vr, Wr)

        rom = LTISystem.from_matrices(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr}

        return rom, rc, reduction_data

    def brbt(self, gamma=1., r=None, tol=None, meth='bfsr'):
        """Reduce using the Bounded Real Balanced Truncation method to order `r` or with tolerance `tol`.

        Parameters
        ----------
        gamma
            Bound for the H_inf norm.
        r
            Order of the reduced model if `tol` is None.
        tol
            Tolerance for the absolute H_inf-error if `r` is None.
        meth
            Method used:
            * square root method ('sr')
            * balancing-free square root method ('bfsr')

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
        assert r is not None and tol is None or r is None and tol is not None
        assert r is None or 0 < r < self.n
        assert meth in {'sr', 'bfsr'}

        self.compute_brcgf(gamma=gamma)
        self.compute_brogf(gamma=gamma)

        if r is not None and r > min([len(self._brcgf), len(self._brogf)]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.\
                              Try reducing the tolerance in the low-rank Lyapunov equation solver.')

        self.compute_brsv_brU_brV(gamma=gamma)

        if r is None:
            bounds = np.cumsum(self._brsv)
            bounds = bounds[-1] - bounds
            bounds *= 2 * self._brgamma
            r = min(i + 1 for i, b in enumerate(bounds) if b <= tol)

        Vr = VectorArrayOperator(self._brcgf).apply(self._brV, ind=range(r))
        Wr = VectorArrayOperator(self._brogf).apply(self._brU, ind=range(r))

        if meth == 'sr':
            alpha = 1 / np.sqrt(self._brsv[:r])
            Vr.scal(alpha)
            Wr.scal(alpha)

            Ar, Br, Cr, Dr, Er = self.project(Vr, Wr, Er_is_identity=True)
        elif meth == 'bfsr':
            Vr = gram_schmidt(Vr, atol=0, rtol=0)
            Wr = gram_schmidt(Wr, atol=0, rtol=0)

            Ar, Br, Cr, Dr, Er = self.project(Vr, Wr)

        rom = LTISystem.from_matrices(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr}

        return rom, rc, reduction_data

    def arnoldi(self, sigma, b_or_c):
        """Rational Arnoldi algorithm

        Implemented only for single-input or single-output systems.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation).
        b_or_c
            Character 'b' or 'c', to choose between the input or output matrix.

        Returns
        -------
        V
            Projection matrix.
        """
        assert b_or_c == 'b' and self.m == 1 or b_or_c == 'c' and self.p == 1

        r = len(sigma)
        V = NumpyVectorArray.make_array(self.n, reserve=r)

        v = NumpyVectorArray(np.array([1.]))
        if b_or_c == 'b':
            v = self.B.apply(v)
        else:
            v = self.C.apply_adjoint(v)
        v.scal(1 / v.l2_norm()[0])

        for i in xrange(r):
            if sigma[i].imag == 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i].real, -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i].real, -1))

                if b_or_c == 'b':
                    v = sEmA.apply_inverse(v)
                else:
                    v = sEmA.apply_inverse_adjoint(v)

                if i > 0:
                    v_norm_orig = v.l2_norm()[0]
                    Vop = VectorArrayOperator(V)
                    v -= Vop.apply(Vop.apply_adjoint(v))
                    if v.l2_norm()[0] < v_norm_orig / 10:
                        v -= Vop.apply(Vop.apply_adjoint(v))
                v.scal(1 / v.l2_norm()[0])
                V.append(v)
            elif sigma[i].imag > 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i], -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i], -1))

                if b_or_c == 'b':
                    v = sEmA.apply_inverse(v)
                else:
                    v = sEmA.apply_inverse_adjoint(v)

                v1 = v.real
                if i > 0:
                    v1_norm_orig = v1.l2_norm()
                    Vop = VectorArrayOperator(V)
                    v1 -= Vop.apply(Vop.apply_adjoint(v1))
                    if v1.l2_norm() < v1_norm_orig / 10:
                        v1 -= Vop.apply(Vop.apply_adjoint(v1))
                v1.scal(1 / v1.l2_norm()[0])
                V.append(v1)

                v2 = v.imag
                v2_norm_orig = v2.l2_norm()
                Vop = VectorArrayOperator(V)
                v2 -= Vop.apply(Vop.apply_adjoint(v2))
                if v2.l2_norm() < v2_norm_orig / 10:
                    v2 -= Vop.apply(Vop.apply_adjoint(v2))
                v2.scal(1 / v2.l2_norm()[0])
                V.append(v2)

                v = v2

        return V

    def arnoldi_tangential(self, sigma, b_or_c, directions):
        """Tangential Rational Arnoldi algorithm

        Implemented only for multi-input or multi-output systems.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation).
        b_or_c
            Character 'b' or 'c', to choose between the input or output matrix.
        directions
            Tangential directions, array of shape (self.m, len(sigma)) or (self.p, len(sigma))

        Returns
        -------
        V
            Projection matrix.
        """
        assert isinstance(directions, np.ndarray)
        r = len(sigma)
        assert (b_or_c == 'b' and self.m > 1 and directions.shape == (self.m, r) or
                b_or_c == 'c' and self.p > 1 and directions.shape == (self.p, r))

        directions = NumpyVectorArray(directions.T)
        directions_norms = directions.l2_norm()
        for i in xrange(r):
            directions.scal(1 / directions_norms[i], ind=i)

        V = NumpyVectorArray.make_array(self.n, reserve=r)

        for i in xrange(r):
            if sigma[i].imag == 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i].real, -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i].real, -1))

                if b_or_c == 'b':
                    if i == 0:
                        v = sEmA.apply_inverse(self.B.apply(directions.real, ind=0))
                    else:
                        VTB = NumpyVectorArray(self.B.apply_adjoint(V).data.T)
                        res = (sEmA.apply(V).data.T.dot(NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse(VTB).data.T) -
                               self.B._matrix.toarray())
                        res = NumpyMatrixOperator(res)
                        v = sEmA.apply_inverse(res.apply(directions.real, ind=i))
                else:
                    if i == 0:
                        v = sEmA.apply_inverse_adjoint(self.C.apply_adjoint(directions.real, ind=0))
                    else:
                        VTCT = NumpyVectorArray(self.C.apply(V).data.T)
                        res = (sEmA.apply_adjoint(V).data.T.dot(NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse_adjoint(VTCT).data.T) -
                               self.C._matrix.T.toarray())
                        res = NumpyMatrixOperator(res)
                        v = sEmA.apply_inverse_adjoint(res.apply(directions.real, ind=i))

                if i > 0:
                    v_norm_orig = v.l2_norm()[0]
                    Vop = VectorArrayOperator(V)
                    v -= Vop.apply(Vop.apply_adjoint(v))
                    if v.l2_norm()[0] < v_norm_orig / 10:
                        v -= Vop.apply(Vop.apply_adjoint(v))
                v.scal(1 / v.l2_norm()[0])
                V.append(v)
            elif sigma[i].imag > 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i], -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i], -1))

                if b_or_c == 'b':
                    if i == 0:
                        v = sEmA.apply_inverse(self.B.apply(directions, ind=0))
                    else:
                        VTB = NumpyVectorArray(self.B.apply_adjoint(V).data.T)
                        res = sEmA.apply(V).data.T.dot(NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse(VTB).data.T) - self.B._matrix.toarray()
                        res = NumpyMatrixOperator(res)
                        v = sEmA.apply_inverse(res.apply(directions, ind=i))
                else:
                    if i == 0:
                        v = sEmA.apply_inverse_adjoint(self.C.apply_adjoint(directions, ind=0))
                    else:
                        VTCT = NumpyVectorArray(self.C.apply(V).data.T)
                        res = (sEmA.apply_adjoint(V).data.T.dot(NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse_adjoint(VTCT).data.T) -
                               self.C._matrix.T.toarray())
                        res = NumpyMatrixOperator(res)
                        v = sEmA.apply_inverse_adjoint(res.apply(directions, ind=i))

                v1 = v.real
                if i > 0:
                    v1_norm_orig = v1.l2_norm()
                    Vop = VectorArrayOperator(V)
                    v1 -= Vop.apply(Vop.apply_adjoint(v1))
                    if v1.l2_norm() < v1_norm_orig / 10:
                        v1 -= Vop.apply(Vop.apply_adjoint(v1))
                v1.scal(1 / v1.l2_norm()[0])
                V.append(v1)

                v2 = v.imag
                v2_norm_orig = v2.l2_norm()
                Vop = VectorArrayOperator(V)
                v2 -= Vop.apply(Vop.apply_adjoint(v2))
                if v2.l2_norm() < v2_norm_orig / 10:
                    v2 -= Vop.apply(Vop.apply_adjoint(v2))
                v2.scal(1 / v2.l2_norm()[0])
                V.append(v2)

                v = v2

        return V

    def interpolation(self, sigma, b, c):
        """Tangential Hermite interpolation at point `sigma` and directions `b` and `c`.

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

        for i in xrange(r):
            b[:, i] /= spla.norm(b[:, i])
            c[:, i] /= spla.norm(c[:, i])

        Vr = NumpyVectorArray.make_array(self.n, reserve=r)
        Wr = NumpyVectorArray.make_array(self.n, reserve=r)

        for i in xrange(r):
            if sigma[i].imag == 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i].real, -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i].real, -1))

                Bb = self.B.apply(NumpyVectorArray(b[:, i].real.T))
                Vr.append(sEmA.apply_inverse(Bb))

                CTc = self.C.apply_adjoint(NumpyVectorArray(c[:, i].real.T))
                Wr.append(sEmA.apply_inverse_adjoint(CTc))
            elif sigma[i].imag > 0:
                if self.E is None:
                    E = NumpyMatrixOperator(sps.eye(self.n, format='csc'))
                    sEmA = self.A.assemble_lincomb((E, self.A), (sigma[i], -1))
                else:
                    sEmA = self.A.assemble_lincomb((self.E, self.A), (sigma[i], -1))

                Bb = self.B.apply(NumpyVectorArray(b[:, i].T))
                v = sEmA.apply_inverse(Bb)
                Vr.append(v.real)
                Vr.append(v.imag)

                CTc = self.C.apply_adjoint(NumpyVectorArray(c[:, i].T))
                w = sEmA.apply_inverse_adjoint(CTc)
                Wr.append(w.real)
                Wr.append(w.imag)

        Vr = gram_schmidt(Vr, atol=0, rtol=0)
        Wr = gram_schmidt(Wr, atol=0, rtol=0)

        return Vr, Wr

    def irka(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_stability=True,
             arnoldi=False):
        """Reduce using IRKA.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation), list of length `r`.
        b
            Initial right tangential directions, |NumPy array| of shape (m, r).
        c
            Initial left tangential directions, |NumPy array| of shape (p, r).
        tol
            Tolerance, largest change in interpolation points.
        maxit
            Maximum number of iterations.
        verbose
            Should consecutive distances be printed.
        force_stability
            If True, new interpolation points are always in the right half-plane.
            Otherwise, they are reflections of reduced order model's poles.
        arnoldi
            Should the Arnoldi process be used for rational interpolation.

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
        assert 0 < r < self.n
        assert sigma is None or len(sigma) == r
        assert b is None or b.shape == (self.m, r)
        assert c is None or c.shape == (self.p, r)

        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            np.random.seed(0)
            b = np.random.randn(self.m, r)
        if c is None:
            np.random.seed(0)
            c = np.random.randn(self.p, r)

        if arnoldi:
            if self.m == 1:
                Vr = self.arnoldi(sigma, 'b')
            else:
                Vr = self.arnoldi_tangential(sigma, 'b', b)
            if self.p == 1:
                Wr = self.arnoldi(sigma, 'c')
            else:
                Wr = self.arnoldi_tangential(sigma, 'c', c)
        else:
            Vr, Wr = self.interpolation(sigma, b, c)

        dist = []
        Sigma = [np.array(sigma)]
        for it in xrange(maxit):
            Ar, Br, Cr, _, Er = self.project(Vr, Wr)

            sigma, Y, X = spla.eig(Ar, Er, left=True, right=True)
            if force_stability:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma = -sigma
            Sigma.append(sigma.copy())

            dist.append([])
            for i in xrange(it + 1):
                dist[-1].append(np.max(np.abs((Sigma[i] - Sigma[-1]) / Sigma[-1])))

            if verbose:
                print('IRKA conv. crit. in step {}: {:.5e}'.format(it + 1, np.min(dist[-1])))

            b = Br.T.dot(Y.conj())
            c = Cr.dot(X)

            if arnoldi:
                if self.m == 1:
                    Vr = self.arnoldi(sigma, 'b')
                else:
                    Vr = self.arnoldi_tangential(sigma, 'b', b)
                if self.p == 1:
                    Wr = self.arnoldi(sigma, 'c')
                else:
                    Wr = self.arnoldi_tangential(sigma, 'c', c)
            else:
                Vr, Wr = self.interpolation(sigma, b, c)

            if np.min(dist[-1]) < tol:
                break

        Ar, Br, Cr, Dr, Er = self.project(Vr, Wr)

        rom = LTISystem.from_matrices(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr, 'dist': dist, 'Sigma': Sigma, 'b': b, 'c': c}

        return rom, rc, reduction_data


class TF(DiscretizationInterface):
    """Class for input-output systems represented by a transfer function

    This class describes input-output systems given by::

        E x'(t) = A x(t) + B u(t)
           y(t) = C x(t) + D u(t)

    if continuous-time, or::

        E x(k + 1) = A x(k) + B u(k)
          y(k)     = C x(k) + D u(k)

    if discrete-time, where A, B, C, D, and E are linear operators.

    Parameters
    ----------
    m
        Number of inputs.
    p
        Number of outputs.
    H
        Transfer function defined at least on the open left complex half-plane.
        H(s) is a |NumPy array| of shape (p, m).
    dH
        Complex derivative of H(s).
    cont_time
        `True` if the system is continuous-time, otherwise discrete-time.
    """
    linear = True

    def __init__(self, m, p, H, dH, cont_time=True):
        assert cont_time in {True, False}

        self.m = m
        self.p = p
        self.H = H
        self.dH = dH
        self.cont_time = cont_time
        self._w = None
        self._tfw = None

    def _solve(self, mu=None):
        raise NotImplementedError('Discretization has no solver.')

    def bode(self, w):
        """Compute the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            Frequencies at which to compute the transfer function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in w, returned as a 3D |NumPy array|
            of shape (p, m, len(w)).
        """
        if not self.cont_time:
            raise NotImplementedError

        self._w = w
        self._tfw = np.empty((self.p, self.m, len(w)), dtype=complex)

        for i, wi in enumerate(w):
            self._tfw[:, :, i] = self.H(1j * wi)

        return self._tfw.copy()

    def interpolation(self, sigma, b, c):
        """Tangential Hermite interpolation at point `sigma` and directions `b` and `c`.

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
        Er
            |NumPy array| of shape (r, r).
        Wr
            Left projection matrix.
        """
        r = len(sigma)

        Er = np.empty((r, r), dtype=complex)
        Ar = np.empty((r, r), dtype=complex)
        Br = np.empty((r, self.m), dtype=complex)
        Cr = np.empty((self.p, r), dtype=complex)

        Ht = np.empty((self.p, self.m, r), dtype=complex)
        dHt = np.empty((self.p, self.m, r), dtype=complex)
        for i in xrange(r):
            Ht[:, :, i] = self.H(sigma[i])
            dHt[:, :, i] = self.dH(sigma[i])

        for i in xrange(r):
            for j in xrange(r):
                if i != j:
                    Er[i, j] = -c[:, i].dot((Ht[:, :, i] -
                               Ht[:, :, j]).dot(b[:, j])) / (sigma[i] - sigma[j])
                    Ar[i, j] = -c[:, i].dot((sigma[i] * Ht[:, :, i] -
                               sigma[j] * Ht[:, :, j])).dot(b[:, j]) / (sigma[i] - sigma[j])
                else:
                    Er[i, i] = -c[:, i].dot(dHt[:, :, i].dot(b[:, j]))
                    Ar[i, i] = -c[:, i].dot((Ht[:, :, i] + sigma[i] * dHt[:, :, i]).dot(b[:, j]))
            Br[i, :] = Ht[:, :, i].T.dot(c[:, i])
            Cr[:, i] = Ht[:, :, i].dot(b[:, i])

        T = np.zeros((r, r), dtype=complex)
        for i in xrange(r):
            if sigma[i].imag == 0:
                T[i, i] = 1
            else:
                try:
                    j = i + 1 + np.where(sigma[i + 1:] == sigma[i].conj)
                except:
                    j = None
                if j:
                    T[i, i] = 1;
                    T[i, j] = 1;
                    T[j, i] = -1j;
                    T[j, j] = 1j;

        Er = (T.dot(Er).dot(T.T)).real
        Ar = (T.dot(Ar).dot(T.T)).real
        Br = (T.dot(Br)).real
        Cr = (Cr.dot(T.T)).real

        return Er, Ar, Br, Cr

    def tf_irka(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_stability=True):
        """Reduce using TF-IRKA.

        Parameters
        ----------
        r
            Order of the reduced order model.
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
        verbose
            Should consecutive distances be printed.
        force_stability
            If True, new interpolation points are always in the right half-plane.
            Otherwise, they are reflections of reduced order model's poles.

        Returns
        -------
        rom
            Reduced |LTISystem| model.
        """
        assert r > 0
        assert sigma is None or len(sigma) == r
        assert b is None or b.shape == (self.m, r)
        assert c is None or c.shape == (self.p, r)

        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            np.random.seed(0)
            b = np.random.randn(self.m, r)
        if c is None:
            np.random.seed(0)
            c = np.random.randn(self.p, r)

        dist = []
        Sigma = [np.array(sigma)]
        for it in xrange(maxit):
            Er, Ar, Br, Cr = self.interpolation(sigma, b, c);

            sigma, Y, X = spla.eig(Ar, Er, left=True, right=True)
            if force_stability:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma = -sigma
            Sigma.append(sigma.copy())

            dist.append([])
            for i in xrange(it + 1):
                dist[-1].append(np.max(np.abs((Sigma[i] - Sigma[-1]) / Sigma[-1])))

            if verbose:
                print('TF-IRKA conv. crit. in step {}: {:.5e}'.format(it + 1, np.min(dist[-1])))

            b = Br.T.dot(Y.conj())
            c = Cr.dot(X)

            if np.min(dist[-1]) < tol:
                break

        Er, Ar, Br, Cr = self.interpolation(sigma, b, c)
        rom = LTISystem.from_matrices(Ar, Br, Cr, None, Er, cont_time=self.cont_time)

        return rom
