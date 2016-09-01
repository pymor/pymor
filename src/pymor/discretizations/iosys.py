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
from pymor.algorithms.riccati import solve_ricc
from pymor.algorithms.to_matrix import to_matrix
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.operators.constructions import (Concatenation, IdentityOperator, LincombOperator,
                                           VectorArrayOperator, ZeroOperator)
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.reductors.basic import GenericRBReconstructor
from pymor.vectorarrays.numpy import NumpyVectorArray


class LTISystem(DiscretizationInterface):
    r"""Class for linear time-invariant systems.

    This class describes input-state-output systems given by

    .. math::
        E x'(t) &= A x(t) + B u(t) \\
           y(t) &= C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + B u(k) \\
          y(k)     &= C x(k) + D u(k)

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
        self.D = D if D is not None else ZeroOperator(B.source, C.range)
        self.E = E if E is not None else IdentityOperator(A.source)
        self.cont_time = cont_time
        self._poles = None
        self._w = None
        self._tfw = None
        self._gramian = {}
        self._sv = {}
        self._U = {}
        self._V = {}
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
            The |NumPy array| or |SciPy spmatrix| A.
        B
            The |NumPy array| or |SciPy spmatrix| B.
        C
            The |NumPy array| or |SciPy spmatrix| C.
        D
            The |NumPy array| or |SciPy spmatrix| D or `None` (then D is assumed to be zero).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (then E is assumed to be the identity).
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

        A = BlockDiagonalOperator((self.A, other.A))
        B = BlockOperator.vstack((self.B, other.B))
        C = BlockOperator.hstack((self.C, other.C))
        D = (self.D + other.D).assemble()
        E = BlockDiagonalOperator((self.E, other.E))

        return LTISystem(A, B, C, D, E, self.cont_time)

    def __neg__(self):
        """Negate |LTISystem|."""
        A = self.A
        B = self.B
        C = (self.C * (-1)).assemble()
        D = (self.D * (-1)).assemble()
        E = self.E

        return LTISystem(A, B, C, D, E, self.cont_time)

    def __sub__(self, other):
        """Subtract two |LTISystems|."""
        return self + (-other)

    def __mul__(self, other):
        """Multiply (cascade) two |LTISystems|."""
        assert self.source == other.range

        A = BlockOperator([[self.A, Concatenation(self.B, other.C)], [None, other.A]])
        B = BlockOperator.vstack((Concatenation(self.B, other.D), other.B))
        C = BlockOperator.hstack((self.C, Concatenation(self.D, other.C)))
        D = Concatenation(self.D, other.D)
        E = BlockDiagonalOperator((self.E, other.E))

        return LTISystem(A, B, C, D, E, self.cont_time)

    def compute_poles(self):
        """Compute system poles."""
        if self._poles is None:
            A = to_matrix(self.A)
            E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E)
            self._poles = spla.eigvals(A, E)

    def eval_tf(self, s):
        """Evaluate the transfer function.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, 2D |NumPy array|.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        sEmA = LincombOperator((E, A), (s, -1))
        if self.m <= self.p:
            I_m = B.source.from_data(sp.eye(self.m))
            tfs = C.apply(sEmA.apply_inverse(B.apply(I_m))).data.T
        else:
            I_p = C.range.from_data(sp.eye(self.p))
            tfs = B.apply_adjoint(sEmA.apply_adjoint_inverse(C.apply_adjoint(I_p))).data
        if not isinstance(D, ZeroOperator):
            if self.m <= self.p:
                tfs += D.apply(I_m).data.T
            else:
                tfs += D.apply_adjoint(I_p).data
        return tfs

    def bode(self, w):
        """Evaluate the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            Frequencies at which to compute the transfer function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in `w`,
            returned as a 3D |NumPy array| of shape `(p, m, len(w))`.
        """
        if not self.cont_time:
            raise NotImplementedError

        self._w = w
        self._tfw = np.dstack([self.eval_tf(1j * wi) for wi in w])
        return self._tfw.copy()

    @classmethod
    def mag_plot(cls, sys_list, plot_style_list=None, w=None, ord=None, dB=False, Hz=False):
        """Draw the magnitude Bode plot.

        Parameters
        ----------
        sys_list
            A single |LTISystem| or a list of |LTISystems|.
        plot_style_list
            A string or a list of strings of the same length as `sys_list`.

            If `None`, matplotlib defaults are used.
        w
            Frequencies at which to compute the transfer function.

            If `None`, use `self._w`.
        ord
            Order of the norm used to compute the magnitude (the default is the Frobenius norm).
        dB
            Should the magnitude be in dB on the plot.
        Hz
            Should the frequency be in Hz on the plot.

        Returns
        -------
        fig
            Matplotlib figure.
        ax
            Matplotlib axes.
        """
        assert isinstance(sys_list, LTISystem) or all(isinstance(sys, LTISystem) for sys in sys_list)
        if isinstance(sys_list, LTISystem):
            sys_list = (sys_list,)

        assert (plot_style_list is None or isinstance(plot_style_list, str) or
                all(isinstance(plot_style, str) for plot_style in plot_style_list))
        if isinstance(plot_style_list, str):
            plot_style_list = (plot_style_list,)

        assert w is not None or all(sys._w is not None for sys in sys_list)

        if w is not None:
            for sys in sys_list:
                sys.bode(w)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, sys in enumerate(sys_list):
            freq = sys._w / (2 * np.pi) if Hz else sys._w
            mag = spla.norm(sys._tfw, ord=ord, axis=(0, 1))
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

    def compute_gramian(self, typ, me_solver=None, tol=None):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            Type of the Gramian:

            * ('lyap', 'cf'): Lyapunov controllability Gramian factor,
            * ('lyap', 'of'): Lyapunov observability Gramian factor,
            * ('lqg', 'cf'): LQG controllability Gramian factor,
            * ('lqg', 'of'): LQG observability Gramian factor,
            * ('br', 'cf', gamma): Bounded Real controllability Gramian factor,
            * ('br', 'of', gamma): Bounded Real observability Gramian factor.
        me_solver
            Matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        tol
            Tolerance parameter for the low-rank matrix equation solver.

            If `None`, then the default tolerance is used. Otherwise, it should be
            a positive float and the Gramian factor is recomputed (if it was already computed).
        """
        assert isinstance(typ, tuple)

        if not self.cont_time:
            raise NotImplementedError

        if typ not in self._gramian or tol is not None:
            A = self.A
            B = self.B
            C = self.C
            E = self.E if not isinstance(self.E, IdentityOperator) else None
            if typ[0] == 'lyap':
                if typ[1] == 'cf':
                    self._gramian[typ] = solve_lyap(A, E, B, trans=False, me_solver=me_solver, tol=tol)
                elif typ[1] == 'of':
                    self._gramian[typ] = solve_lyap(A, E, C, trans=True, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError('Only cf and of are possible for lyap Gramian.')
            elif typ[0] == 'lqg':
                if typ[1] == 'cf':
                    self._gramian[typ] = solve_ricc(A, E=E, B=B, C=C, trans=True, me_solver=me_solver, tol=tol)
                elif typ[1] == 'of':
                    self._gramian[typ] = solve_ricc(A, E=E, B=B, C=C, trans=False, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError('Only cf and of are possible for lqg Gramian.')
            elif typ[0] == 'br':
                assert isinstance(typ[2], float) and typ[2] > 0
                if typ[1] == 'cf':
                    self._gramian[typ] = solve_ricc(A, E=E, B=B / np.sqrt(typ[2]), C=C / np.sqrt(typ[2]),
                                                    R=-IdentityOperator(self.C.range),
                                                    trans=True, me_solver=me_solver, tol=tol)
                elif typ[1] == 'of':
                    self._gramian[typ] = solve_ricc(A, E=E, B=B / np.sqrt(typ[2]), C=C / np.sqrt(typ[2]),
                                                    R=-IdentityOperator(self.B.source),
                                                    trans=False, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError('Only cf and of are possible for br Gramian.')
            else:
                raise NotImplementedError('Only lyap, lqg, and br Gramians are available.')

    def compute_sv_U_V(self, typ, me_solver=None):
        """Compute singular values and vectors.

        Parameters
        ----------
        typ
            Type of the Gramian:

            * ('lyap',): Lyapunov Gramian,
            * ('lqg',): LQG Gramian,
            * ('br', gamma): Bounded Real Gramian,
        me_solver
            Matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        """
        assert isinstance(typ, tuple)

        if typ not in self._sv or typ not in self._U or typ not in self._V:
            typ_cf = (typ[0], 'cf') + typ[1:]
            typ_of = (typ[0], 'of') + typ[1:]
            self.compute_gramian(typ_cf, me_solver=me_solver)
            self.compute_gramian(typ_of, me_solver=me_solver)

            U, self._sv[typ], Vh = spla.svd(self.E.apply2(self._gramian[typ_of], self._gramian[typ_cf]))

            self._U[typ] = NumpyVectorArray(U.T)
            self._V[typ] = NumpyVectorArray(Vh)

    def norm(self, name='H2'):
        """Compute a norm of the |LTISystem|.

        Parameters
        ----------
        name
            Name of the norm ('H2', 'Hinf', 'Hankel').
        """
        if name == 'H2':
            if self._H2_norm is not None:
                return self._H2_norm

            if ('lyap', 'cf') in self._gramian:
                self._H2_norm = np.sqrt(self.C.apply(self._gramian[('lyap', 'cf')]).l2_norm2().sum())
            elif ('lyap', 'of') in self._gramian:
                self._H2_norm = np.sqrt(self.B.apply_adjoint(self._gramian[('lyap', 'of')]).l2_norm2().sum())
            elif self.m <= self.p:
                self.compute_gramian(('lyap', 'cf'))
                self._H2_norm = np.sqrt(self.C.apply(self._gramian[('lyap', 'cf')]).l2_norm2().sum())
            else:
                self.compute_gramian(('lyap', 'of'))
                self._H2_norm = np.sqrt(self.B.apply_adjoint(self._gramian[('lyap', 'of')]).l2_norm2().sum())

            return self._H2_norm
        elif name == 'Hinf':
            if self._Hinf_norm is not None:
                return self._Hinf_norm

            from slycot import ab13dd
            dico = 'C' if self.cont_time else 'D'
            jobe = 'I' if self.E is None else 'G'
            equil = 'S'
            jobd = 'Z' if self.D is None else 'D'
            A, B, C, D, E = map(to_matrix, (self.A, self.B, self.C, self.D, self.E))
            self._Hinf_norm, self._fpeak = ab13dd(dico, jobe, equil, jobd, self.n, self.m, self.p, A, E, B, C, D)

            return self._Hinf_norm
        elif name == 'Hankel':
            self.compute_sv_U_V(('lyap',))
            return self._sv[('lyap',)][0]
        else:
            raise NotImplementedError('Only H2, Hinf, and Hankel norms are implemented.')

    def project(self, Vr, Wr, Er_is_identity=False):
        """Reduce using Petrov-Galerkin projection.

        Parameters
        ----------
        Vr
            Right projection matrix, |VectorArray| from `self.A.source`.
        Wr
            Left projection matrix, |VectorArray| from `self.A.source`.
        Er_is_identity
            If the reduced `E` is guaranteed to be the identity matrix.

        Returns
        -------
        Ar
            Reduced `A` operator, |NumpyMatrixOperator|.
        Br
            Reduced `B` operator, :class:`~pymor.operators.constructions.VectorArrayOperator` (transposed).
        Cr
            Reduced `C` operator, :class:`~pymor.operators.constructions.VectorArrayOperator`.
        Dr
            Simply `self.D`.
        Er
            Reduced `E` operator, |NumpyMatrixOperator| or `None`.
        """
        Ar = NumpyMatrixOperator(self.A.apply2(Wr, Vr))
        Br = VectorArrayOperator(self.B.apply_adjoint(Wr), transposed=True)
        Cr = VectorArrayOperator(self.C.apply(Vr))
        Dr = self.D

        if Er_is_identity:
            Er = None
        else:
            Er = NumpyMatrixOperator(self.E.apply2(Wr, Vr))

        return Ar, Br, Cr, Dr, Er

    def bt(self, r=None, tol=None, typ=('lyap',), me_solver=None, method='bfsr'):
        """Reduce using the Balanced Truncation method to order `r` or with tolerance `tol`.

        .. [A05]  A. Antoulas, Approximation of Large-Scale Dynamical Systems,
                  SIAM, 2005.
        .. [MG91] D. Mustafa, K. Glover, Controller Reduction by H_∞-Balanced Truncation,
                  IEEE Transactions on Automatic Control, 36(6), 668-682, 1991.
        .. [OJ88] P. C. Opdenacker, E. A. Jonckheere, A Contraction Mapping
                  Preserving Balanced Reduction Scheme and Its Infinity Norm
                  Error Bounds,
                  IEEE Transactions on Circuits and Systems, 35(2), 184-189, 1988.

        Parameters
        ----------
        r
            Order of the reduced model if `tol` is `None`.
        tol
            Tolerance for the error bound if `r` is `None`.
        typ
            Type of the Gramian (see :func:`~pymor.discretizations.iosys.LTISystem.compute_sv_U_V`).
        me_solver
            Matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        method
            Projection method used:

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
            Dictionary of additional data produced by the reduction process.
            Contains projection matrices `Vr` and `Wr`.
        """
        assert r is not None and tol is None or r is None and tol is not None
        assert r is None or 0 < r < self.n
        assert method in ('sr', 'bfsr')

        typ_cf = (typ[0], 'cf') + typ[1:]
        typ_of = (typ[0], 'of') + typ[1:]
        self.compute_gramian(typ_cf, me_solver=me_solver)
        self.compute_gramian(typ_of, me_solver=me_solver)

        if r is not None and r > min([len(self._gramian[typ_cf]), len(self._gramian[typ_of])]):
            raise ValueError('r needs to be smaller than the sizes of Gramian factors.' +
                             ' Try reducing the tolerance in the low-rank Lyapunov equation solver.')

        self.compute_sv_U_V(typ, me_solver=me_solver)

        if r is None:
            bounds = np.zeros(self._sv[typ].shape)
            if typ[0] == 'lyap':
                bounds[:-1] = 2 * self._sv[typ][-1:0:-1].cumsum()[::-1]
            elif typ[0] == 'lqg':
                tmp = self._sv[typ][-1:0:-1]
                bounds[:-1] = 2 * (tmp / np.sqrt(1 + tmp ** 2)).cumsum()[::-1]
            elif typ[0] == 'br':
                bounds[:-1] = 2 * typ[1] * self._sv[typ][-1:0:-1].cumsum()[::-1]
            r = np.argmax(bounds <= tol) + 1

        Vr = VectorArrayOperator(self._gramian[typ_cf]).apply(self._V[typ], ind=list(range(r)))
        Wr = VectorArrayOperator(self._gramian[typ_of]).apply(self._U[typ], ind=list(range(r)))

        if method == 'sr':
            alpha = 1 / np.sqrt(self._sv[typ][:r])
            Vr.scal(alpha)
            Wr.scal(alpha)
            Ar, Br, Cr, Dr, Er = self.project(Vr, Wr, Er_is_identity=True)
        elif method == 'bfsr':
            Vr = gram_schmidt(Vr, atol=0, rtol=0)
            Wr = gram_schmidt(Wr, atol=0, rtol=0)
            Ar, Br, Cr, Dr, Er = self.project(Vr, Wr)

        rom = LTISystem(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr}

        return rom, rc, reduction_data

    def arnoldi(self, sigma, b_or_c):
        """Rational Arnoldi algorithm.

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
        V = self.A.source.type.make_array(self.A.source.subtype, reserve=r)

        v = np.array([1.])
        if b_or_c == 'b':
            v = self.B.source.from_data(v)
            v = self.B.apply(v)
        else:
            v = self.C.range.from_data(v)
            v = self.C.apply_adjoint(v)
        v.scal(1 / v.l2_norm()[0])

        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = LincombOperator((self.E, self.A), (sigma[i].real, -1))

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
                sEmA = LincombOperator((self.E, self.A), (sigma[i], -1))

                if b_or_c == 'b':
                    v = sEmA.apply_inverse(v)
                else:
                    v = sEmA.apply_inverse_adjoint(v)

                v1 = v.real
                if i > 0:
                    v1_norm_orig = v1.l2_norm()[0]
                    Vop = VectorArrayOperator(V)
                    v1 -= Vop.apply(Vop.apply_adjoint(v1))
                    if v1.l2_norm()[0] < v1_norm_orig / 10:
                        v1 -= Vop.apply(Vop.apply_adjoint(v1))
                v1.scal(1 / v1.l2_norm()[0])
                V.append(v1)

                v2 = v.imag
                v2_norm_orig = v2.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v2 -= Vop.apply(Vop.apply_adjoint(v2))
                if v2.l2_norm()[0] < v2_norm_orig / 10:
                    v2 -= Vop.apply(Vop.apply_adjoint(v2))
                v2.scal(1 / v2.l2_norm()[0])
                V.append(v2)

                v = v2

        return V

    def arnoldi_tangential(self, sigma, b_or_c, directions):
        """Tangential Rational Arnoldi algorithm.

        Implemented only for multi-input or multi-output systems.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation).
        b_or_c
            Character 'b' or 'c', to choose between the input or output matrix.
        directions
            Tangential directions, |VectorArray| of length `len(sigma)` from `self.B.source`
            or `self.C.range` (depending on `b_or_c`).

        Returns
        -------
        V
            Projection matrix.
        """
        r = len(sigma)
        assert len(directions) == r
        assert (b_or_c == 'b' and self.m > 1 and directions in self.B.source or
                b_or_c == 'c' and self.p > 1 and directions in self.C.range)

        directions.scal(1 / directions.l2_norm())

        V = self.A.source.type.make_array(self.A.source.subtype, reserve=r)

        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = LincombOperator((self.E, self.A), (sigma[i].real, -1))

                if b_or_c == 'b':
                    if i == 0:
                        v = sEmA.apply_inverse(self.B.apply(directions.real, ind=0))
                    else:
                        Bd = self.B.apply(directions.real, ind=i)
                        VTBd = VectorArrayOperator(V, transposed=True).apply(Bd)
                        sEmA_proj_inv_VTBd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse(VTBd)
                        V_sEmA_proj_inv_VTBd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTBd)
                        rd = sEmA.apply(V_sEmA_proj_inv_VTBd) - Bd
                        v = sEmA.apply_inverse(rd)
                else:
                    if i == 0:
                        v = sEmA.apply_inverse_adjoint(self.C.apply_adjoint(directions.real, ind=0))
                    else:
                        CTd = self.C.apply_adjoint(directions.real, ind=i)
                        VTCTd = VectorArrayOperator(V, transposed=True).apply(CTd)
                        sEmA_proj_inv_VTCTd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse_adjoint(VTCTd)
                        V_sEmA_proj_inv_VTCTd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTCTd)
                        rd = sEmA.apply_adjoint(V_sEmA_proj_inv_VTCTd) - CTd
                        v = sEmA.apply_inverse_adjoint(rd)

                if i > 0:
                    v_norm_orig = v.l2_norm()[0]
                    Vop = VectorArrayOperator(V)
                    v -= Vop.apply(Vop.apply_adjoint(v))
                    if v.l2_norm()[0] < v_norm_orig / 10:
                        v -= Vop.apply(Vop.apply_adjoint(v))
                v.scal(1 / v.l2_norm()[0])
                V.append(v)
            elif sigma[i].imag > 0:
                sEmA = LincombOperator((self.E, self.A), (sigma[i], -1))

                if b_or_c == 'b':
                    if i == 0:
                        v = sEmA.apply_inverse(self.B.apply(directions, ind=0))
                    else:
                        Bd = self.B.apply(directions, ind=i)
                        VTBd = VectorArrayOperator(V, transposed=True).apply(Bd)
                        sEmA_proj_inv_VTBd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse(VTBd)
                        V_sEmA_proj_inv_VTBd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTBd)
                        rd = sEmA.apply(V_sEmA_proj_inv_VTBd) - Bd
                        v = sEmA.apply_inverse(rd)
                else:
                    if i == 0:
                        v = sEmA.apply_inverse_adjoint(self.C.apply_adjoint(directions, ind=0))
                    else:
                        CTd = self.C.apply_adjoint(directions, ind=i)
                        VTCTd = VectorArrayOperator(V, transposed=True).apply(CTd)
                        sEmA_proj_inv_VTCTd = NumpyMatrixOperator(sEmA.apply2(V, V)).apply_inverse_adjoint(VTCTd)
                        V_sEmA_proj_inv_VTCTd = VectorArrayOperator(V).apply(sEmA_proj_inv_VTCTd)
                        rd = sEmA.apply_adjoint(V_sEmA_proj_inv_VTCTd) - CTd
                        v = sEmA.apply_inverse_adjoint(rd)

                v1 = v.real
                if i > 0:
                    v1_norm_orig = v1.l2_norm()[0]
                    Vop = VectorArrayOperator(V)
                    v1 -= Vop.apply(Vop.apply_adjoint(v1))
                    if v1.l2_norm()[0] < v1_norm_orig / 10:
                        v1 -= Vop.apply(Vop.apply_adjoint(v1))
                v1.scal(1 / v1.l2_norm()[0])
                V.append(v1)

                v2 = v.imag
                v2_norm_orig = v2.l2_norm()[0]
                Vop = VectorArrayOperator(V)
                v2 -= Vop.apply(Vop.apply_adjoint(v2))
                if v2.l2_norm()[0] < v2_norm_orig / 10:
                    v2 -= Vop.apply(Vop.apply_adjoint(v2))
                v2.scal(1 / v2.l2_norm()[0])
                V.append(v2)

                v = v2

        return V

    def interpolation(self, sigma, b, c):
        """Tangential Hermite interpolation at points `sigma` and directions `b` and `c`.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of length `r`.
        b
            Right tangential directions, |VectorArray| of length `r` from `self.B.source`.
        c
            Left tangential directions, |VectorArray| of length `r` from `self.C.range`.

        Returns
        -------
        Vr
            Right projection matrix, |VectorArray| from `self.A.source`.
        Wr
            Left projection matrix, |VectorArray| from `self.A.source`.
        """
        r = len(sigma)
        assert b in self.B.source and len(b) == r
        assert c in self.C.range and len(c) == r

        b.scal(1 / b.l2_norm())
        c.scal(1 / c.l2_norm())

        Vr = self.A.source.type.make_array(self.A.source.subtype, reserve=r)
        Wr = self.A.source.type.make_array(self.A.source.subtype, reserve=r)

        for i in range(r):
            if sigma[i].imag == 0:
                sEmA = LincombOperator((self.E, self.A), (sigma[i].real, -1))

                Bb = self.B.apply(b.real, ind=i)
                Vr.append(sEmA.apply_inverse(Bb))

                CTc = self.C.apply_adjoint(c.real, ind=i)
                Wr.append(sEmA.apply_inverse_adjoint(CTc))
            elif sigma[i].imag > 0:
                sEmA = LincombOperator((self.E, self.A), (sigma[i], -1))

                Bb = self.B.apply(b, ind=i)
                v = sEmA.apply_inverse(Bb)
                Vr.append(v.real)
                Vr.append(v.imag)

                CTc = self.C.apply_adjoint(c, ind=i)
                w = sEmA.apply_inverse_adjoint(CTc)
                Wr.append(w.real)
                Wr.append(w.imag)

        Vr = gram_schmidt(Vr, atol=0, rtol=0)
        Wr = gram_schmidt(Wr, atol=0, rtol=0)

        return Vr, Wr

    def irka(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_sigma_in_rhp=True,
             arnoldi=False, compute_errors=False):
        """Reduce using IRKA.

        .. [GAB08] S. Gugercin, A. C. Antoulas, C. A. Beattie, H_2 model reduction
                   for large-scale linear dynamical systems,
                   SIAM Journal on Matrix Analysis and Applications, 30(2), 609-638, 2008.
        .. [ABG10] A. C. Antoulas, C. A. Beattie, S. Gugercin, Interpolatory model
                   reduction of large-scale dynamical systems,
                   Efficient Modeling and Control of Large-Scale Systems, Springer-Verlag, 2010.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation), list of length `r`.

            If `None`, interpolation points are log-spaced between 0.1 and 10.
        b
            Initial right tangential directions, |VectorArray| of length `r` from `self.B.source`.

            If `None`, `b` is chosen with all ones.
        c
            Initial left tangential directions, |VectorArray| of length `r` from `self.C.range`.

            If `None`, `c` is chosen with all ones.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        verbose
            Should consecutive distances be printed.
        force_sigma_in_rhp
            If True, new interpolation points are always in the right half-plane.
            Otherwise, they are reflections of reduced order model's poles.
        arnoldi
            Should the Arnoldi process be used for rational interpolation.
        compute_errors
            Should the relative :math:`\mathcal{H}_2`-errors of intermediate reduced order models be computed.

        Returns
        -------
        rom
            Reduced |LTISystem| model.
        rc
            Reconstructor of full state.
        reduction_data
            Dictionary of additional data produced by the reduction process. Contains:

            * projection matrices `Vr` and `Wr`,
            * distances between interpolation points in different iterations `dist`,
            * interpolation points from all iterations `Sigma`,
            * right and left tangential directions `R` and `L`, and
            * relative :math:`\mathcal{H}_2`-errors `errors` (if compute_errors is `True`).
        """
        assert 0 < r < self.n
        assert sigma is None or len(sigma) == r
        assert b is None or b in self.B.source and len(b) == r
        assert c is None or c in self.C.range and len(c) == r

        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            b = self.B.source.from_data(np.ones((r, self.m)))
        if c is None:
            c = self.C.range.from_data(np.ones((r, self.p)))

        if verbose:
            if compute_errors:
                print('iter | shift change | rel. H_2-error')
                print('-----+--------------+---------------')
            else:
                print('iter | shift change')
                print('-----+-------------')

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
        R = [b]
        L = [c]
        if compute_errors:
            errors = []
        for it in range(maxit):
            Ar, Br, Cr, _, Er = self.project(Vr, Wr)

            if compute_errors:
                rom = LTISystem(Ar, Br, Cr, None, Er, cont_time=self.cont_time)
                err = self - rom
                try:
                    rel_H2_err = err.norm() / self.norm()
                except:
                    rel_H2_err = np.inf
                errors.append(rel_H2_err)

            sigma, Y, X = spla.eig(Ar._matrix, Er._matrix, left=True, right=True)
            if force_sigma_in_rhp:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma *= -1
            Sigma.append(sigma)

            dist.append([])
            for i in range(it + 1):
                dist[-1].append(np.max(np.abs((Sigma[i] - Sigma[-1]) / Sigma[-1])))

            if verbose:
                if compute_errors:
                    print('{:4d} | {:.6e} | {:.6e}'.format(it + 1, np.min(dist[-1]), rel_H2_err))
                else:
                    print('{:4d} | {:.6e}'.format(it + 1, np.min(dist[-1])))

            b = Br.apply_adjoint(NumpyVectorArray(Y.conj().T))
            c = Cr.apply(NumpyVectorArray(X.T))
            R.append(b)
            L.append(c)

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

        rom = LTISystem(Ar, Br, Cr, Dr, Er, cont_time=self.cont_time)
        rc = GenericRBReconstructor(Vr)
        reduction_data = {'Vr': Vr, 'Wr': Wr, 'dist': dist, 'Sigma': Sigma, 'R': R, 'L': L}
        if compute_errors:
            reduction_data['errors'] = errors

        return rom, rc, reduction_data


class TF(DiscretizationInterface):
    """Class for input-output systems represented by a transfer function.

    This class describes input-output systems given by a transfer function :math:`H(s)`.

    Parameters
    ----------
    m
        Number of inputs.
    p
        Number of outputs.
    H
        Transfer function defined at least on the open right complex half-plane.

        H(s) is a |NumPy array| of shape `(p, m)`.
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
            of shape `(p, m, len(w))`.
        """
        if not self.cont_time:
            raise NotImplementedError

        self._w = w
        self._tfw = np.dstack([self.H(1j * wi) for wi in w])

        return self._tfw.copy()

    def interpolation(self, sigma, b, c):
        """Tangential Hermite interpolation at point `sigma` and directions `b` and `c`.

        Parameters
        ----------
        sigma
            Interpolation points (closed under conjugation), list of length `r`.
        b
            Right tangential directions, |NumPy array| of shape `(self.m, r)`.
        c
            Left tangential directions, |NumPy array| of shape `(self.p, r)`.

        Returns
        -------
        Er
            |NumPy array| of shape `(r, r)`.
        Ar
            |NumPy array| of shape `(r, r)`.
        Br
            |NumPy array| of shape `(r, self.m)`.
        Cr
            |NumPy array| of shape `(self.p, r)`.
        """
        r = len(sigma)
        assert b.shape == (self.m, r)
        assert c.shape == (self.p, r)

        for i in range(r):
            b[:, i] /= spla.norm(b[:, i])
            c[:, i] /= spla.norm(c[:, i])

        Er = np.empty((r, r), dtype=complex)
        Ar = np.empty((r, r), dtype=complex)
        Br = np.empty((r, self.m), dtype=complex)
        Cr = np.empty((self.p, r), dtype=complex)

        Ht = np.dstack([self.H(s) for s in sigma])
        dHt = np.dstack([self.dH(s) for s in sigma])

        for i in range(r):
            for j in range(r):
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
        for i in range(r):
            if sigma[i].imag == 0:
                T[i, i] = 1
            else:
                try:
                    j = i + 1 + np.where(np.isclose(sigma[i + 1:], sigma[i].conjugate()))[0][0]
                except:
                    j = None
                if j:
                    T[i, i] = 1
                    T[i, j] = 1
                    T[j, i] = -1j
                    T[j, j] = 1j

        Er = (T.dot(Er).dot(T.conj().T)).real
        Ar = (T.dot(Ar).dot(T.conj().T)).real
        Br = (T.dot(Br)).real
        Cr = (Cr.dot(T.conj().T)).real

        return Er, Ar, Br, Cr

    def tf_irka(self, r, sigma=None, b=None, c=None, tol=1e-4, maxit=100, verbose=False, force_sigma_in_rhp=True):
        """Reduce using TF-IRKA.

        .. [AG12] C. A. Beattie, S. Gugercin, Realization-independent H2-approximation,
                  Proceedings of the 51st IEEE Conference on Decision and Control, 2012.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation), list of length `r`.

            If `None`, interpolation points are log-spaced between 0.1 and 10.
        b
            Initial right tangential directions, |NumPy array| of shape `(self.m, r)`.

            If `None`, `b` is chosen with all ones.
        c
            Initial left tangential directions, |NumPy array| of shape `(self.p, r)`.

            If `None`, `c` is chosen with all ones.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        verbose
            Should consecutive distances be printed.
        force_sigma_in_rhp
            If True, new interpolation points are always in the right half-plane.
            Otherwise, they are reflections of reduced order model's poles.

        Returns
        -------
        rom
            Reduced |LTISystem| model.
        reduction_data
            Dictionary of additional data produced by the reduction process. Contains:

            * distances between interpolation points in different iterations `dist`,
            * interpolation points from all iterations `Sigma`, and
            * right and left tangential directions `R` and `L`.
        """
        assert r > 0
        assert sigma is None or len(sigma) == r
        assert b is None or b.shape == (self.m, r)
        assert c is None or c.shape == (self.p, r)

        if sigma is None:
            sigma = np.logspace(-1, 1, r)
        if b is None:
            b = np.ones((self.m, r))
        if c is None:
            c = np.ones((self.p, r))

        if verbose:
            print('iter | shift change')
            print('-----+-------------')

        dist = []
        Sigma = [np.array(sigma)]
        R = [b]
        L = [c]
        for it in range(maxit):
            Er, Ar, Br, Cr = self.interpolation(sigma, b, c)

            sigma, Y, X = spla.eig(Ar, Er, left=True, right=True)
            if force_sigma_in_rhp:
                sigma = np.array([np.abs(s.real) + s.imag * 1j for s in sigma])
            else:
                sigma *= -1
            Sigma.append(sigma)

            dist.append([])
            for i in range(it + 1):
                dist[-1].append(np.max(np.abs((Sigma[i] - Sigma[-1]) / Sigma[-1])))

            if verbose:
                print('{:4d} | {:.6e}'.format(it + 1, np.min(dist[-1])))

            b = Br.T.dot(Y.conj())
            c = Cr.dot(X)
            R.append(b)
            L.append(c)

            if np.min(dist[-1]) < tol:
                break

        Er, Ar, Br, Cr = self.interpolation(sigma, b, c)
        rom = LTISystem.from_matrices(Ar, Br, Cr, None, Er, cont_time=self.cont_time)
        reduction_data = {'dist': dist, 'Sigma': Sigma, 'R': R, 'L': L}

        return rom, reduction_data
