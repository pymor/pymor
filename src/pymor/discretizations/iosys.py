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
from pymor.core.cache import cached
from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.block import BlockOperator, BlockDiagonalOperator
from pymor.operators.constructions import Concatenation, IdentityOperator, LincombOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator


class InputOutputSystem(DiscretizationBase):
    """Base class for input-output systems."""

    def __init__(self, m, p, cont_time=True, cache_region='memory', name=None, **kwargs):
        self.m = m
        self.p = p
        super().__init__(cache_region=cache_region, name=name, **kwargs)
        self.cont_time = cont_time

    def _solve(self, mu=None):
        raise NotImplementedError

    def __add__(self, other):
        """Add two input-state-output systems."""
        assert type(self) == type(other)
        assert self.cont_time == other.cont_time

        def add_operator(op, other_op):
            if op.source.id == 'INPUT':
                if op.range.id == 'OUTPUT':
                    return BlockOperator([[(op + other_op).assemble()]], source_id=op.source.id, range_id=op.range.id)
                elif op.range.id == 'STATE':
                    return BlockOperator.vstack((op, other_op), source_id=op.source.id, range_id=op.range.id)
                else:
                    raise NotImplementedError
            else:
                if op.range.id == 'OUTPUT':
                    return BlockOperator.hstack((op, other_op), source_id=op.source.id, range_id=op.range.id)
                elif op.range.id == 'STATE':
                    return BlockDiagonalOperator((op, other_op), source_id=op.source.id, range_id=op.range.id)
                else:
                    raise NotImplementedError

        new_operators = {k: add_operator(self.operators[k], other.operators[k]) for k in self.operators}

        return self.with_(operators=new_operators, cont_time=self.cont_time)

    def __neg__(self):
        """Negate input-state-output system."""
        new_operators = {k: (op * (-1)).assemble() if op.range.id == 'OUTPUT' else op
                         for k, op in self.operators.items()}

        return self.with_(operators=new_operators)

    def __sub__(self, other):
        """Subtract two input-state-output system."""
        return self + (-other)

    def eval_tf(self, s, mu=None):
        """Evaluate the transfer function of the system."""
        raise NotImplementedError

    def eval_dtf(self, s, mu=None):
        """Evaluate the derivative of the transfer function of the system."""
        raise NotImplementedError

    @cached
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
            |NumPy array| of shape `(self.p, self.m, len(w))`.
        """
        if not self.cont_time:
            raise NotImplementedError

        return np.dstack([self.eval_tf(1j * wi) for wi in w])

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
        ord
            The order of the norm used to compute the magnitude (the default is
            the Frobenius norm).
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
        assert isinstance(sys_list, InputOutputSystem) or all(isinstance(sys, InputOutputSystem) for sys in sys_list)
        if isinstance(sys_list, LTISystem):
            sys_list = (sys_list,)

        assert (plot_style_list is None or isinstance(plot_style_list, str) and len(sys_list) == 1 or
                all(isinstance(plot_style, str) for plot_style in plot_style_list) and
                len(sys_list) == len(plot_style_list))
        if isinstance(plot_style_list, str):
            plot_style_list = (plot_style_list,)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, sys in enumerate(sys_list):
            tfw = sys.bode(w)
            freq = w / (2 * np.pi) if Hz else w
            mag = spla.norm(tfw, ord=ord, axis=(0, 1))
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
        return fig, ax


class LTISystem(InputOutputSystem):
    r"""Class for linear time-invariant systems.

    This class describes input-state-output systems given by

    .. math::
        E x'(t) &= A x(t) + B u(t) \\
           y(t) &= C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + B u(k) \\
          y(k)     &= C x(k) + D u(k)

    if discrete-time, where :math:`A`, :math:`B`, :math:`C`, :math:`D`, and
    :math:`E` are linear operators.

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
        The |Operator| E or `None` (then E is assumed to be identity).
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    n
        The order of the system.
    A
        The |Operator| A.
    B
        The |Operator| B.
    C
        The |Operator| C.
    D
        The |Operator| D.
    E
        The |Operator| E.
    operators
        Dict of all |Operators| appearing in the discretization.
    """

    linear = True

    special_operators = frozenset({'A', 'B', 'C', 'D', 'E'})

    def __init__(self, A, B, C, D=None, E=None, cont_time=True, cache_region='memory', name=None):

        D = D or ZeroOperator(B.source, C.range)
        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range and A.source.id == 'STATE'
        assert B.linear and B.range == A.source and B.source.id == 'INPUT'
        assert C.linear and C.source == A.range and C.range.id == 'OUTPUT'
        assert D.linear and D.source == B.source and D.range == C.range
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(B.source.dim, C.range.dim, cont_time=cont_time, cache_region=cache_region, name=name,
                         A=A, B=B, C=C, D=D, E=E)

        self.n = A.source.dim

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
            The |NumPy array| or |SciPy spmatrix| D or `None` (then D is assumed
            to be zero).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (then E is assumed
            to be identity).
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        assert isinstance(A, (np.ndarray, sps.spmatrix))
        assert isinstance(B, (np.ndarray, sps.spmatrix))
        assert isinstance(C, (np.ndarray, sps.spmatrix))
        assert D is None or isinstance(D, (np.ndarray, sps.spmatrix))
        assert E is None or isinstance(E, (np.ndarray, sps.spmatrix))

        A = NumpyMatrixOperator(A, source_id='STATE', range_id='STATE')
        B = NumpyMatrixOperator(B, source_id='INPUT', range_id='STATE')
        C = NumpyMatrixOperator(C, source_id='STATE', range_id='OUTPUT')
        if D is not None:
            D = NumpyMatrixOperator(D, source_id='INPUT', range_id='OUTPUT')
        if E is not None:
            E = NumpyMatrixOperator(E, source_id='STATE', range_id='STATE')

        return cls(A, B, C, D, E, cont_time=cont_time)

    @classmethod
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, cont_time=True):
        """Create |LTISystem| from matrices stored in separate files.

        Parameters
        ----------
        A_file
            The name of the file (with extension) containing A.
        B_file
            The name of the file (with extension) containing B.
        C_file
            The name of the file (with extension) containing C.
        D_file
            `None` or the name of the file (with extension) containing D.
        E_file
            `None` or the name of the file (with extension) containing E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix

        A = load_matrix(A_file)
        B = load_matrix(B_file)
        C = load_matrix(C_file)
        D = load_matrix(D_file) if D_file is not None else None
        E = load_matrix(E_file) if E_file is not None else None

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time)

    @classmethod
    def from_mat_file(cls, file_name, cont_time=True):
        """Create |LTISystem| from matrices stored in a .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be
            included) containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        import scipy.io as spio
        mat_dict = spio.loadmat(file_name)

        assert 'A' in mat_dict and 'B' in mat_dict and 'C' in mat_dict

        A = mat_dict['A']
        B = mat_dict['B']
        C = mat_dict['C']
        D = mat_dict['D'] if 'D' in mat_dict else None
        E = mat_dict['E'] if 'E' in mat_dict else None

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time)

    @classmethod
    def from_abcde_files(cls, files_basename, cont_time=True):
        """Create |LTISystem| from matrices stored in a .[ABCDE] files.

        Parameters
        ----------
        files_basename
            The basename of files containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.

        Returns
        -------
        lti
            The |LTISystem| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix
        import os.path

        A = load_matrix(files_basename + '.A')
        B = load_matrix(files_basename + '.B')
        C = load_matrix(files_basename + '.C')
        D = load_matrix(files_basename + '.D') if os.path.isfile(files_basename + '.D') else None
        E = load_matrix(files_basename + '.E') if os.path.isfile(files_basename + '.E') else None

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time)

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

        return self.__class__(A, B, C, D, E, cont_time=self.cont_time)

    @cached
    def poles(self):
        """Compute system poles."""
        A = to_matrix(self.A)
        E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E)
        return spla.eigvals(A, E)

    def eval_tf(self, s):
        """Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C (s E - A)^{-1} B + D.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array|
            of shape `(self.p, self.m)`.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        sEmA = LincombOperator((E, A), (s, -1))
        if self.m <= self.p:
            tfs = C.apply(sEmA.apply_inverse(B.as_range_array())).data.T
        else:
            tfs = B.apply_adjoint(sEmA.apply_inverse_adjoint(C.as_source_array())).data.conj()
        if not isinstance(D, ZeroOperator):
            if self.m <= self.p:
                tfs += D.as_range_array().data.T
            else:
                tfs += D.as_source_array().data
        return tfs

    def eval_dtf(self, s):
        """Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C (s E - A)^{-1} E (s E - A)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        A = self.A
        B = self.B
        C = self.C
        E = self.E

        sEmA = LincombOperator((E, A), (s, -1))
        if self.m <= self.p:
            dtfs = -C.apply(sEmA.apply_inverse(E.apply(sEmA.apply_inverse(B.as_range_array())))).data.T
        else:
            dtfs = B.apply_adjoint(sEmA.apply_inverse_adjoint(E.apply_adjoint(sEmA.apply_inverse_adjoint(
                C.as_source_array())))).data.conj()
        return dtfs

    @cached
    def gramian(self, typ, subtyp, me_solver=None, tol=None):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'lyap'`: Lyapunov Gramian,
            - `'lqg'`: LQG Gramian,
            - `('br', gamma)`: Bounded Real Gramian with parameter gamma.
        subtyp
            The subtype of the Gramian:

            - `'cf'`: controllability Gramian factor,
            - `'of'`: observability Gramian factor.
        me_solver
            The matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        tol
            The tolerance parameter for the low-rank matrix equation solver.

            If `None`, then the default tolerance is used. Otherwise, it should
            be a positive float and the Gramian factor is recomputed (if it was
            already computed).
        """
        assert isinstance(typ, (str, tuple))
        assert isinstance(subtyp, str)

        if not self.cont_time:
            raise NotImplementedError

        A = self.A
        B = self.B
        C = self.C
        E = self.E if not isinstance(self.E, IdentityOperator) else None

        if typ == 'lyap':
            if subtyp == 'cf':
                return solve_lyap(A, E, B, trans=False, me_solver=me_solver, tol=tol)
            elif subtyp == 'of':
                return solve_lyap(A, E, C, trans=True, me_solver=me_solver, tol=tol)
            else:
                raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for 'lyap' type.")
        elif typ == 'lqg':
            if subtyp == 'cf':
                return solve_ricc(A, E=E, B=B, C=C, trans=True, me_solver=me_solver, tol=tol)
            elif subtyp == 'of':
                return solve_ricc(A, E=E, B=B, C=C, trans=False, me_solver=me_solver, tol=tol)
            else:
                raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for 'lqg' type.")
        elif isinstance(typ, tuple) and typ[0] == 'br':
            assert isinstance(typ[1], float)
            assert typ[1] > 0
            c = 1 / np.sqrt(typ[1])
            if subtyp == 'cf':
                return solve_ricc(A, E=E, B=B * c, C=C * c, R=IdentityOperator(C.range) * (-1),
                                  trans=True, me_solver=me_solver, tol=tol)
            elif subtyp == 'of':
                return solve_ricc(A, E=E, B=B * c, C=C * c, R=IdentityOperator(B.source) * (-1),
                                  trans=False, me_solver=me_solver, tol=tol)
            else:
                raise NotImplementedError("Only 'cf' and 'of' subtypes are possible for ('br', gamma) type.")
        else:
            raise NotImplementedError("Only 'lyap', 'lqg', and ('br', gamma) types are available.")

    @cached
    def sv_U_V(self, typ, me_solver=None):
        """Compute singular values and vectors.

        Parameters
        ----------
        typ
            The type of the Gramian (see
            :func:`~pymor.discretizations.iosys.LTISystem.gramian`).
        me_solver
            The matrix equation solver to use (see
            :func:`pymor.algorithms.lyapunov.solve_lyap` or
            :func:`pymor.algorithms.riccati.solve_ricc`).
        """
        cf = self.gramian(typ, 'cf', me_solver=me_solver)
        of = self.gramian(typ, 'of', me_solver=me_solver)

        U, sv, Vh = spla.svd(self.E.apply2(of, cf))
        return sv, U.T, Vh

    @cached
    def norm(self, name='H2', me_solver=None):
        """Compute a norm of the |LTISystem|.

        Parameters
        ----------
        name
            The name of the norm (`'H2'`, `'Hinf'`, `'Hinf_fpeak'`, `'Hankel'`).
        """
        if name == 'H2':
            B, C = self.B, self.C

            if self.m <= self.p:
                cf = self.gramian('lyap', 'cf', me_solver=me_solver)
                return np.sqrt(C.apply(cf).l2_norm2().sum())
            else:
                of = self.gramian('lyap', 'of', me_solver=me_solver)
                return np.sqrt(B.apply_adjoint(of).l2_norm2().sum())

        elif name == 'Hinf_fpeak':
            from slycot import ab13dd
            dico = 'C' if self.cont_time else 'D'
            jobe = 'I' if isinstance(self.E, IdentityOperator) else 'G'
            equil = 'S'
            jobd = 'Z' if isinstance(self.D, ZeroOperator) else 'D'
            A, B, C, D, E = map(to_matrix, (self.A, self.B, self.C, self.D, self.E))
            Hinf, fpeak = ab13dd(dico, jobe, equil, jobd, self.n, self.m, self.p, A, E, B, C, D)
            return Hinf, fpeak
        elif name == 'Hinf':
            return self.norm('Hinf_fpeak', me_solver=me_solver)[0]
        elif name == 'Hankel':
            return self.sv_U_V('lyap', me_solver=me_solver)[0][0]
        else:
            raise NotImplementedError('Only H2, Hinf, and Hankel norms are implemented.')


class TF(InputOutputSystem):
    """Class for input-output systems represented by a transfer function.

    This class describes input-output systems given by a transfer function
    :math:`H(s)`.

    Parameters
    ----------
    m
        The number of inputs.
    p
        The number of outputs.
    H
        The transfer function defined at least on the open right complex
        half-plane.

        `H(s)` is a |NumPy array| of shape `(p, m)`.
    dH
        The complex derivative of `H`.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    tf
        The transfer function.
    dtf
        The complex derivative of the transfer function.
    """
    linear = True

    def __init__(self, m, p, H, dH, cont_time=True, cache_region='memory', name=None):
        assert cont_time in (True, False)

        self.tf = H
        self.dtf = dH
        super().__init__(m=m, p=p, cont_time=cont_time, cache_region=cache_region, name=name)

    def eval_tf(self, s):
        return self.tf(s)

    def eval_dtf(self, s):
        return self.dtf(s)


class SecondOrderSystem(InputOutputSystem):
    r"""Class for linear second order systems.

    This class describes input-output systems given by

    .. math::
        M x''(t) + D x'(t) + K x(t) &= B u(t) \\
                               y(t) &= C_p x(t) + C_v x'(t)

    if continuous-time, or

    .. math::
        M x(k + 2) + D x(k + 1) + K x(k) &= B u(k) \\
                                    y(k) &= C_p x(k) + C_v x(k + 1)

    if discrete-time, where :math:`M`, :math:`D`, :math:`K`, :math:`B`,
    :math:`C_p`, and :math:`C_v` are linear operators.

    Parameters
    ----------
    M
        The |Operator| M or `None` (then M is assumed to be identity).
    D
        The |Operator| D or `None` (then D is assumed to be zero).
    K
        The |Operator| K.
    B
        The |Operator| B.
    Cp
        The |Operator| Cp.
    Cv
        The |Operator| Cv.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    n
        The order of the system (equal to M.source.dim).
    M
        The |Operator| M.
    D
        The |Operator| D.
    K
        The |Operator| K.
    B
        The |Operator| B.
    Cp
        The |Operator| Cp.
    Cv
        The |Operator| Cv.
    operators
        Dictionary of all |Operators| contained in the discretization.
    """
    linear = True

    special_operators = frozenset({'M', 'D', 'K', 'B', 'Cp', 'Cv'})

    def __init__(self, K, B, Cp, Cv, M=None, D=None,
                 cont_time=True, cache_region='memory', name=None):

        M = M or IdentityOperator(K.source)
        D = D or ZeroOperator(K.source, K.range)

        assert K.linear and K.source == K.range and K.source.id == 'STATE'
        assert B.linear and B.range == K.source and B.source.id == 'INPUT'
        assert Cp.linear and Cp.source == K.range and Cp.range.id == 'OUTPUT'
        assert Cv.linear and Cv.source == K.range
        assert Cp.range == Cv.range
        assert M.linear and M.source == M.range == K.source
        assert D.linear and D.source == D.range == K.source
        assert cont_time in (True, False)

        super().__init__(B.source.dim, Cp.range.dim, cont_time=cont_time, cache_region=cache_region, name=name,
                         K=K, B=B, Cp=Cp, Cv=Cv, M=M, D=D)

        self.n = K.source.dim

    def eval_tf(self, s):
        """Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            (C_p + s C_v) (s^2 M + s D + K)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array|
            of shape `(self.p, self.m)`.
        """
        M = self.M
        D = self.D
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv

        s2MpsDpK = LincombOperator((M, D, K), (s ** 2, s, 1))
        if self.m <= self.p:
            CppsCv = LincombOperator((Cp, Cv), (1, s))
            tfs = CppsCv.apply(s2MpsDpK.apply_inverse(B.as_range_array())).data.T
        else:
            tfs = B.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(Cp.as_source_array() +
                                                                 Cv.as_source_array() * s.conj())).data.conj()
        return tfs

    def eval_dtf(self, s):
        """Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            s C_v (s^2 M + s D + K)^{-1} B
            - (C_p + s C_v) (s^2 M + s D + K)^{-1} (2 s M + D)
                (s^2 M + s D + K)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        M = self.M
        D = self.D
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv

        s2MpsDpK = LincombOperator((M, D, K), (s ** 2, s, 1))
        sM2pD = LincombOperator((M, D), (2 * s, 1))
        if self.m <= self.p:
            dtfs = Cv.apply(s2MpsDpK.apply_inverse(B.as_range_array())).data.T * s
            CppsCv = LincombOperator((Cp, Cv), (1, s))
            dtfs -= CppsCv.apply(s2MpsDpK.apply_inverse(sM2pD.apply(s2MpsDpK.apply_inverse(B.as_range_array())))).data.T
        else:
            dtfs = B.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(Cv.as_source_array())).data.conj() * s
            dtfs -= B.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(sM2pD.apply_adjoint(s2MpsDpK.apply_inverse_adjoint(
                Cp.as_source_array() + Cv.as_source_array() * s.conj())))).data.conj()
        return dtfs


class LinearDelaySystem(InputOutputSystem):
    r"""Class for linear delay systems.

    This class describes input-output systems given by

    .. math::
        E x'(t) &= A x(t) + \sum_{i = 1}^q{A_i x(t - \tau_i)} + B u(t) \\
           y(t) &= C x(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + \sum_{i = 1}^q{A_i x(k - \tau_i)} + B u(k) \\
              y(k) &= C x(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`, :math:`B`,
    and :math:`C` are linear operators.

    Parameters
    ----------
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    A
        The |Operator| A.
    Ad
        The tuple of |Operators| A_i.
    tau
        The tuple of delay times (positive floats or ints).
    B
        The |Operator| B.
    C
        The |Operator| C.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    q
        The number of delay terms.
    E
        The |Operator| E.
    A
        The |Operator| A.
    Ad
        The tuple of |Operators| A_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    operators
        Dict of all |Operators| appearing in the discretization.
    """
    linear = True
    special_operators = frozenset({'A', 'Ad', 'B', 'C', 'E'})

    def __init__(self, A, Ad, tau, B, C, E=None, cont_time=True, cache_region='memory', name=None):

        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range and A.source.id == 'STATE'
        assert isinstance(Ad, tuple) and len(Ad) > 0
        assert all(Ai.linear and Ai.source == Ai.range == A.source for Ai in Ad)
        assert isinstance(tau, tuple) and len(tau) == len(Ad) and all(taui > 0 for taui in tau)
        assert B.linear and B.range == A.source and B.source.id == 'INPUT'
        assert C.linear and C.source == A.range and C.range.id == 'OUTPUT'
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(m=B.source.dim, p=C.range.dim, cont_time=cont_time, cache_region=cache_region, name=name,
                         A=A, Ad=Ad, B=B, C=C, E=E)
        self.tau = tau
        self.n = A.source.dim
        self.q = len(Ad)

    def eval_tf(self, s):
        r"""Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C \left(s E - A - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array|
            of shape `(self.p, self.m)`.
        """
        E = self.E
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C

        middle = LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        if self.m <= self.p:
            tfs = C.apply(middle.apply_inverse(B.as_range_array())).data.T
        else:
            tfs = B.apply_adjoint(middle.apply_inverse_adjoint(C.as_source_array())).data.conj()
        return tfs

    def eval_dtf(self, s):
        r"""Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C \left(s E - A - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1}
                \left(E + \sum_{i = 1}^q{\tau_i e^{-\tau_i s} A_i}\right)
                \left(s E - A - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of outputs
            is small compared to the order of the system, e.g. less than 10.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        E = self.E
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C

        left_and_right = LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        middle = LincombOperator((E,) + Ad, (s,) + tuple(taui * np.exp(-taui * s) for taui in self.tau))
        if self.m <= self.p:
            dtfs = C.apply(left_and_right.apply_inverse(middle.apply(left_and_right.apply_inverse(
                B.as_range_array())))).data.T
        else:
            dtfs = B.apply_adjoint(left_and_right.apply_inverse_adjoint(middle.apply_adjoint(
                left_and_right.apply_inverse_adjointi(C.as_source_array())))).data.conj()
        return dtfs


class LinearStochasticSystem(InputOutputSystem):
    r"""Class for linear stochastic systems.

    This class describes input-output systems given by

    .. math::
        E \mathrm{d}x(t) &= A x(t) \mathrm{d}t
                            + \sum_{i = 1}^q{A_i x(t) \mathrm{d}\omega_i(t)}
                            + B u(t) \mathrm{d}t \\
                    y(t) &= C x(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + \sum_{i = 1}^q{A_i x(k) \omega_i(k)} + B u(k) \\
              y(k) &= C x(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`, :math:`B`,
    and :math:`C` are linear operators and :math:`\omega_i` are stochastic
    processes.

    Parameters
    ----------
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    A
        The |Operator| A.
    As
        The tuple of |Operators| A_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    q
        The number of stochastic processes.
    E
        The |Operator| E.
    A
        The |Operator| A.
    As
        The tuple of |Operators| A_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    operators
        Dictionary of all |Operators| contained in the discretization.
    """

    linear = True
    special_operators = frozenset({'A', 'As', 'B', 'C', 'E'})

    def __init__(self, A, As, tau, B, C, E=None, cont_time=True, cache_region='memory', name=None):

        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range and A.source == 'STATE'
        assert isinstance(As, tuple) and len(As) > 0
        assert all(Ai.linear and Ai.source == Ai.range == A.source for Ai in As)
        assert isinstance(tau, tuple) and len(tau) == len(As) and all(taui > 0 for taui in tau)
        assert B.linear and B.range == A.source and B.source.id == 'INPUT'
        assert C.linear and C.source == A.range and C.range.id == 'OUTPUT'
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(m=B.source.dim, p=C.range.dim, cont_time=cont_time, cache_region=cache_region, name=name,
                         A=A, As=As, B=B, C=C, E=E)

        self.n = A.source.dim
        self.q = len(As)
        self.tau = tau


class BilinearSystem(InputOutputSystem):
    r"""Class for bilinear systems.

    This class describes input-output systems given by

    .. math::
        E x'(t) &= A x(t) + \sum_{i = 1}^m{N_i x(t) u_i(t)} + B u(t) \\
           y(t) &= C x(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + \sum_{i = 1}^m{N_i x(k) u_i(k)} + B u(k) \\
              y(k) &= C x(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`N_i`, :math:`B`,
    and :math:`C` are linear operators and :math:`m` is the number of
    inputs.

    Parameters
    ----------
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    A
        The |Operator| A.
    N
        The tuple of |Operators| N_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    E
        The |Operator| E.
    A
        The |Operator| A.
    N
        The tuple of |Operators| N_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    operators
        Dictionary of all |Operators| contained in the discretization.
    """

    linear = False

    def __init__(self, A, N, B, C, E=None, cont_time=True, cache_region='memory', name=None):

        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range and A.source.id == 'STATE'
        assert B.linear and B.range == A.source and B.source.id == 'INPUT'
        assert isinstance(N, tuple) and len(N) == B.source.dim
        assert all(Ni.linear and Ni.source == Ni.range == A.source for Ni in N)
        assert C.linear and C.source == A.range and C.range.id == 'OUTPUT'
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(m=B.source.dim, p=C.range.dim, cont_time=cont_time, cache_region=cache_region, name=name,
                         A=A, N=N, B=B, C=C, E=E)

        self.n = A.source.dim
