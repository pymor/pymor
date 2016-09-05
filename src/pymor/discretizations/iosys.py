# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.lyapunov import solve_lyap
from pymor.algorithms.riccati import solve_ricc
from pymor.algorithms.to_matrix import to_matrix
from pymor.discretizations.interfaces import DiscretizationInterface
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.operators.constructions import (Concatenation, IdentityOperator, LincombOperator, ZeroOperator)
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.frozendict import FrozenDict
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
    ss_operators
        Dictonary for state-to-state operators A and E.
    is_operators
        Dictonary for input-to-state operator B.
    so_operators
        Dictonary for state-to-output operator C.
    io_operators
        Dictonary for input-to-output operator D.
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
    A
        The |Operator| A. The same as `ss_operators['A']`.
    B
        The |Operator| B. The same as `is_operators['B']`.
    C
        The |Operator| C. The same as `so_operators['C']`.
    D
        The |Operator| D. The same as `io_operators['D']`.
    E
        The |Operator| E. The same as `ss_operators['E']`.
    """
    linear = True

    def __init__(self, A=None, B=None, C=None, D=None, E=None, ss_operators=None, is_operators=None,
                 so_operators=None, io_operators=None, cont_time=True):
        A = A or ss_operators['A']
        B = B or is_operators['B']
        C = C or so_operators['C']
        D = D or io_operators.get('D')
        E = E or ss_operators.get('E')
        assert isinstance(A, OperatorInterface) and A.linear
        assert A.source == A.range
        assert isinstance(B, OperatorInterface) and B.linear
        assert B.range == A.source
        assert isinstance(C, OperatorInterface) and C.linear
        assert C.source == A.range
        assert (D is None or
                isinstance(D, OperatorInterface) and D.linear and D.source == B.source and D.range == C.range)
        assert E is None or isinstance(E, OperatorInterface) and E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        self.n = A.source.dim
        self.m = B.source.dim
        self.p = C.range.dim
        self.A = A
        self.B = B
        self.C = C
        self.D = D if D is not None else ZeroOperator(B.source, C.range)
        self.E = E if E is not None else IdentityOperator(A.source)
        self.ss_operators = FrozenDict({'A': A, 'E': self.E})
        self.si_operators = FrozenDict({'B': B})
        self.os_operators = FrozenDict({'C': C})
        self.oi_operators = FrozenDict({'D': self.D})
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
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, cont_time=True):
        """Create |LTISystem| from matrices stored in separate files.

        Parameters
        ----------
        A_file
            Name of the file (with extension) containing A.
        B_file
            Name of the file (with extension) containing B.
        C_file
            Name of the file (with extension) containing C.
        D_file
            `None` or name of the file (with extension) containing D.
        E_file
            `None` or name of the file (with extension) containing E.
        cont_time
            `True` if the system is continuous-time, otherwise discrete-time.

        Returns
        -------
        lti
            |LTISystem| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix

        A = load_matrix(A_file)
        B = load_matrix(B_file)
        C = load_matrix(C_file)
        D = load_matrix(D_file) if D_file is not None else None
        E = load_matrix(E_file) if E_file is not None else None

        return cls.from_matrices(A, B, C, D, E, cont_time)

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
        D = mat_dict['D'] if 'D' in mat_dict else None
        E = mat_dict['E'] if 'E' in mat_dict else None

        return cls.from_matrices(A, B, C, D, E, cont_time)

    @classmethod
    def from_abcde_files(cls, files_basename, cont_time=True):
        """Create |LTISystem| from matrices stored in a .[ABCDE] files.

        Parameters
        ----------
        files_basename
            Basename of files containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise discrete-time.

        Returns
        -------
        lti
            |LTISystem| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix
        import os.path

        A = load_matrix(files_basename + '.A')
        B = load_matrix(files_basename + '.B')
        C = load_matrix(files_basename + '.C')
        D = load_matrix(files_basename + '.D') if os.path.isfile(files_basename + '.D') else None
        E = load_matrix(files_basename + '.E') if os.path.isfile(files_basename + '.E') else None

        return cls.from_matrices(A, B, C, D, E, cont_time)

    def _solve(self, mu=None):
        raise NotImplementedError('Discretization has no solver.')

    def __add__(self, other):
        """Add two |LTISystems|."""
        assert isinstance(other, LTISystem)
        assert self.B.source == other.B.source
        assert self.C.range == other.C.range
        assert self.cont_time == other.cont_time

        A = BlockDiagonalOperator((self.A, other.A))
        B = BlockOperator.vstack((self.B, other.B))
        C = BlockOperator.hstack((self.C, other.C))
        D = (self.D + other.D).assemble()
        E = BlockDiagonalOperator((self.E, other.E))

        return self.__class__(A, B, C, D, E, self.cont_time)

    def __neg__(self):
        """Negate |LTISystem|."""
        A = self.A
        B = self.B
        C = (self.C * (-1)).assemble()
        D = (self.D * (-1)).assemble()
        E = self.E

        return self.__class__(A, B, C, D, E, self.cont_time)

    def __sub__(self, other):
        """Subtract two |LTISystems|."""
        return self + (-other)

    def __mul__(self, other):
        """Multiply (cascade) two |LTISystems|."""
        assert self.B.source == other.C.range

        A = BlockOperator([[self.A, Concatenation(self.B, other.C)],
                           [None, other.A]])
        B = BlockOperator.vstack((Concatenation(self.B, other.D),
                                  other.B))
        C = BlockOperator.hstack((self.C, Concatenation(self.D,
                                  other.C)))
        D = Concatenation(self.D, other.D)
        E = BlockDiagonalOperator((self.E, other.E))

        return self.__class__(A, B, C, D, E, self.cont_time)

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

    def compute_gramian(self, typ, subtyp, me_solver=None, tol=None):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            Type of the Gramian:

            - `'lyap'`: Lyapunov Gramian,
            - `'lqg'`: LQG Gramian,
            - `('br', gamma)`: Bounded Real Gramian with parameter gamma.
        subtyp
            Subtype of the Gramian:

            - `'cf'`: controllability Gramian factor,
            - `'of'`: observability Gramian factor.
        me_solver
            Matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        tol
            Tolerance parameter for the low-rank matrix equation solver.

            If `None`, then the default tolerance is used. Otherwise, it should be
            a positive float and the Gramian factor is recomputed (if it was already computed).
        """
        assert isinstance(typ, (str, tuple))
        assert isinstance(subtyp, str)

        if not self.cont_time:
            raise NotImplementedError

        if typ not in self._gramian or subtyp not in self._gramian[typ] or tol is not None:
            A = self.A
            B = self.B
            C = self.C
            E = self.E if not isinstance(self.E, IdentityOperator) else None
            if typ == 'lyap':
                if subtyp == 'cf':
                    self._gramian[typ][subtyp] = solve_lyap(A, E, B, trans=False, me_solver=me_solver, tol=tol)
                elif subtyp == 'of':
                    self._gramian[typ][subtyp] = solve_lyap(A, E, C, trans=True, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for 'lyap' type.")
            elif typ == 'lqg':
                if subtyp == 'cf':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B, C=C, trans=True, me_solver=me_solver, tol=tol)
                elif subtyp == 'of':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B, C=C, trans=False, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for 'lqg' type.")
            elif isinstance(typ, tuple) and typ[0] == 'br':
                assert isinstance(typ[1], float)
                assert typ[1] > 0
                if subtyp == 'cf':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B / np.sqrt(typ[1]), C=C / np.sqrt(typ[1]),
                                                            R=-IdentityOperator(C.range),
                                                            trans=True, me_solver=me_solver, tol=tol)
                elif subtyp == 'of':
                    self._gramian[typ][subtyp] = solve_ricc(A, E=E, B=B / np.sqrt(typ[1]), C=C / np.sqrt(typ[1]),
                                                            R=-IdentityOperator(B.source),
                                                            trans=False, me_solver=me_solver, tol=tol)
                else:
                    raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for ('br', gamma) type.")
            else:
                raise NotImplementedError("Only 'lyap', 'lqg', and ('br', gamma) types are available.")

    def compute_sv_U_V(self, typ, me_solver=None):
        """Compute singular values and vectors.

        Parameters
        ----------
        typ
            Type of the Gramian (see :func:`~pymor.discretizations.iosys.LTISystem.compute_gramian`).
        me_solver
            Matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        """
        assert isinstance(typ, tuple)

        if typ not in self._sv or typ not in self._U or typ not in self._V:
            self.compute_gramian(typ, 'cf', me_solver=me_solver)
            self.compute_gramian(typ, 'of', me_solver=me_solver)

            U, self._sv[typ], Vh = spla.svd(self.E.apply2(self._gramian[typ]['of'], self._gramian[typ]['cf']))

            self._U[typ] = NumpyVectorArray(U.T)
            self._V[typ] = NumpyVectorArray(Vh)

    def norm(self, name='H2'):
        """Compute a norm of the |LTISystem|.

        Parameters
        ----------
        name
            Name of the norm (`'H2'`, `'Hinf'`, `'Hankel'`).
        """
        if name == 'H2':
            if self._H2_norm is not None:
                return self._H2_norm

            B, C = self.B, self.C

            if 'lyap' in self._gramian and 'cf' in self._gramian['lyap']:
                self._H2_norm = np.sqrt(C.apply(self._gramian['lyap']['cf']).l2_norm2().sum())
            elif 'lyap' in self._gramian and 'of' in self._gramian['lyap']:
                self._H2_norm = np.sqrt(B.apply_adjoint(self._gramian['lyap']['of']).l2_norm2().sum())
            elif self.m <= self.p:
                self.compute_gramian('lyap', 'cf')
                self._H2_norm = np.sqrt(C.apply(self._gramian['lyap']['cf']).l2_norm2().sum())
            else:
                self.compute_gramian('lyap', 'of')
                self._H2_norm = np.sqrt(B.apply_adjoint(self._gramian['lyap']['of']).l2_norm2().sum())

            return self._H2_norm
        elif name == 'Hinf':
            if self._Hinf_norm is not None:
                return self._Hinf_norm

            from slycot import ab13dd
            dico = 'C' if self.cont_time else 'D'
            jobe = 'I' if isinstance(self.E, IdentityOperator) else 'G'
            equil = 'S'
            jobd = 'Z' if isinstance(self.D, ZeroOperator) else 'D'
            A, B, C, D, E = map(to_matrix, (self.A, self.B, self.C,
                                            self.D, self.E))
            self._Hinf_norm, self._fpeak = ab13dd(dico, jobe, equil, jobd, self.n, self.m, self.p, A, E, B, C, D)

            return self._Hinf_norm
        elif name == 'Hankel':
            self.compute_sv_U_V('lyap')
            return self._sv['lyap'][0]
        else:
            raise NotImplementedError('Only H2, Hinf, and Hankel norms are implemented.')


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

        `H(s)` is a |NumPy array| of shape `(p, m)`.
    dH
        Complex derivative of `H`.
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
            Transfer function values at frequencies in `w`, returned as a 3D |NumPy array|
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

        .. [AG12] C. A. Beattie, S. Gugercin, Realization-independent
                  H2-approximation,
                  Proceedings of the 51st IEEE Conference on Decision and
                  Control, 2012.

        Parameters
        ----------
        r
            Order of the reduced order model.
        sigma
            Initial interpolation points (closed under conjugation), list of
            length `r`.

            If `None`, interpolation points are log-spaced between 0.1 and
            10.
        b
            Initial right tangential directions, |NumPy array| of shape
            `(self.m, r)`.

            If `None`, `b` is chosen with all ones.
        c
            Initial left tangential directions, |NumPy array| of shape
            `(self.p, r)`.

            If `None`, `c` is chosen with all ones.
        tol
            Tolerance for the largest change in interpolation points.
        maxit
            Maximum number of iterations.
        verbose
            Should consecutive distances be printed.
        force_sigma_in_rhp
            If `True`, new interpolation points are always in the right
            half-plane. Otherwise, they are reflections of reduced order
            model's poles.

        Returns
        -------
        rom
            Reduced |LTISystem| model.
        reduction_data
            Dictionary of additional data produced by the reduction process.
            Contains:

            - distances between interpolation points in different iterations
              `dist`,
            - interpolation points from all iterations `Sigma`, and
            - right and left tangential directions `R` and `L`.
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
