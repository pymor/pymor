# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from functools import partial

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import cached
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.discretizations.basic import DiscretizationBase
from pymor.operators.block import (BlockOperator, BlockRowOperator, BlockColumnOperator, BlockDiagonalOperator,
                                   SecondOrderSystemOperator)
from pymor.operators.constructions import Concatenation, IdentityOperator, LincombOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.block import BlockVectorSpace


class InputOutputSystem(DiscretizationBase):
    """Base class for input-output systems."""

    def __init__(self, input_space, output_space, cont_time=True, cache_region='memory', name=None, **kwargs):
        self.input_space = input_space
        self.output_space = output_space
        super().__init__(cache_region=cache_region, name=name, **kwargs)
        self.cont_time = cont_time

    @property
    def m(self):
        return self.input_space.dim

    @property
    def p(self):
        return self.output_space.dim

    def _solve(self, mu=None):
        raise NotImplementedError

    def eval_tf(self, s, mu=None):
        """Evaluate the transfer function."""
        raise NotImplementedError

    def eval_dtf(self, s, mu=None):
        """Evaluate the derivative of the transfer function."""
        raise NotImplementedError

    @cached
    def bode(self, w):
        """Evaluate the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            Angular frequencies at which to compute the transfer
            function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in `w`,
            |NumPy array| of shape `(len(w), self.p, self.m)`.
        """
        if not self.cont_time:
            raise NotImplementedError

        return np.stack([self.eval_tf(1j * wi) for wi in w])

    def mag_plot(self, w, ax=None, ord=None, Hz=False, dB=False, **mpl_kwargs):
        """Draw the magnitude Bode plot.

        Parameters
        ----------
        w
            Angular frequencies at which to evaluate the transfer
            function.
        ax
            Axis to which to plot.
            If not given, `matplotlib.pyplot.gca` is used.
        ord
            The order of the norm used to compute the magnitude (the
            default is the Frobenius norm).
        Hz
            Should the frequency be in Hz on the plot.
        dB
            Should the magnitude be in dB on the plot.
        mpl_kwargs
            Keyword arguments used in the matplotlib plot function.

        Returns
        -------
        out
            List of matplotlib artists added.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        freq = w / (2 * np.pi) if Hz else w
        mag = spla.norm(self.bode(w), ord=ord, axis=(1, 2))
        if dB:
            out = ax.semilogx(freq, 20 * np.log2(mag), **mpl_kwargs)
        else:
            out = ax.loglog(freq, mag, **mpl_kwargs)

        ax.set_title('Magnitude Bode Plot')
        freq_unit = ' (Hz)' if Hz else ' (rad/s)'
        ax.set_xlabel('Frequency' + freq_unit)
        mag_unit = ' (dB)' if dB else ''
        ax.set_ylabel('Magnitude' + mag_unit)

        return out


class InputStateOutputSystem(InputOutputSystem):
    """Base class for input-output systems with state space."""

    def __init__(self, input_space, state_space, output_space, cont_time=True, cache_region='memory', name=None, **kwargs):
        # ensure that state_space can be distinguished from input and output space
        # ensure that ids are different to make sure that also reduced spaces can be differentiated
        assert state_space.id != input_space.id and state_space.id != output_space.id
        super().__init__(input_space, output_space, cont_time=cont_time, cache_region=cache_region, name=name, **kwargs)
        self.state_space = state_space

    @property
    def n(self):
        return self.state_space.dim

    def __add__(self, other):
        """Add two input-state-output systems."""
        assert type(self) == type(other)
        assert self.cont_time == other.cont_time

        def add_operator(op, other_op):
            if op.source == self.input_space:
                if op.range == self.output_space:
                    return (op + other_op).assemble()
                elif op.range == self.state_space:
                    return BlockColumnOperator([op, other_op], range_id=self.state_space.id)
                else:
                    raise NotImplementedError
            elif op.source == self.state_space:
                if op.range == self.output_space:
                    return BlockRowOperator([op, other_op], source_id=self.state_space.id)
                elif op.range == self.state_space:
                    if isinstance(op, IdentityOperator) and isinstance(other_op, IdentityOperator):
                        return IdentityOperator(BlockVectorSpace([op.source, other_op.source], self.state_space.id))
                    else:
                        return BlockDiagonalOperator([op, other_op], source_id=self.state_space.id,
                                                     range_id=self.state_space.id)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        new_operators = {k: add_operator(self.operators[k], other.operators[k]) for k in self.operators}

        return self.with_(operators=new_operators, cont_time=self.cont_time)

    def __neg__(self):
        """Negate input-state-output system."""
        new_operators = {k: (op * (-1)).assemble() if op.range == self.output_space else op
                         for k, op in self.operators.items()}

        return self.with_(operators=new_operators)

    def __sub__(self, other):
        """Subtract two input-state-output system."""
        return self + (-other)


_DEFAULT_ME_SOLVER_BACKEND = 'pymess' if config.HAVE_PYMESS else \
                             'slycot' if config.HAVE_SLYCOT else \
                             'scipy'


class LTISystem(InputStateOutputSystem):
    r"""Class for linear time-invariant systems.

    This class describes input-state-output systems given by

    .. math::
        E x'(t) &= A x(t) + B u(t) \\
           y(t) &= C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1) &= A x(k) + B u(k) \\
          y(k)     &= C x(k) + D u(k)

    if discrete-time, where :math:`A`, :math:`B`, :math:`C`, :math:`D`,
    and :math:`E` are linear operators.

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
    solver_options
        The solver options to use to solve the Lyapunov equations.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call
        `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
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

    special_operators = frozenset({'A', 'B', 'C', 'D', 'E'})

    def __init__(self, A, B, C, D=None, E=None, cont_time=True,
                 solver_options=None, estimator=None, visualizer=None,
                 cache_region='memory', name=None):

        D = D or ZeroOperator(C.range, B.source)
        E = E or IdentityOperator(A.source)

        assert A.linear
        assert A.source == A.range
        assert B.linear
        assert B.range == A.source
        assert C.linear
        assert C.source == A.range
        assert D.linear
        assert D.source == B.source
        assert D.range == C.range
        assert E.linear
        assert E.source == E.range
        assert E.source == A.source
        assert cont_time in (True, False)
        assert solver_options is None or solver_options.keys() <= {'lyap', 'ricc'}

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name,
                         A=A, B=B, C=C, D=D, E=E)

        self.solution_space = A.source
        self.solver_options = solver_options

    @classmethod
    def from_matrices(cls, A, B, C, D=None, E=None, cont_time=True,
                      input_id='INPUT', state_id='STATE', output_id='OUTPUT',
                      solver_options=None, estimator=None, visualizer=None,
                      cache_region='memory', name=None):
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
            The |NumPy array| or |SciPy spmatrix| D or `None` (then D is
            assumed to be zero).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (then E is
            assumed to be identity).
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, discretization)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            discretization which will call
            `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, discretization, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the discretization which forwards its arguments to the
            visualizer's `visualize` method.
        cache_region
            `None` or name of the cache region to use. See
            :mod:`pymor.core.cache`.
        name
            Name of the system.

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

        A = NumpyMatrixOperator(A, source_id=state_id, range_id=state_id)
        B = NumpyMatrixOperator(B, source_id=input_id, range_id=state_id)
        C = NumpyMatrixOperator(C, source_id=state_id, range_id=output_id)
        if D is not None:
            D = NumpyMatrixOperator(D, source_id=input_id, range_id=output_id)
        if E is not None:
            E = NumpyMatrixOperator(E, source_id=state_id, range_id=state_id)

        return cls(A, B, C, D, E, cont_time=cont_time,
                   solver_options=solver_options, estimator=estimator, visualizer=visualizer,
                   cache_region=cache_region, name=name)

    @classmethod
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, cont_time=True,
                   solver_options=None, estimator=None, visualizer=None,
                   cache_region='memory', name=None):
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
            `None` or the name of the file (with extension) containing
            D.
        E_file
            `None` or the name of the file (with extension) containing
            E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, discretization)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            discretization which will call
            `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, discretization, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the discretization which forwards its arguments to the
            visualizer's `visualize` method.
        cache_region
            `None` or name of the cache region to use. See
            :mod:`pymor.core.cache`.
        name
            Name of the system.

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

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time,
                                 solver_options=solver_options, estimator=estimator, visualizer=visualizer,
                                 cache_region=cache_region, name=name)

    @classmethod
    def from_mat_file(cls, file_name, cont_time=True,
                      solver_options=None, estimator=None, visualizer=None,
                      cache_region='memory', name=None):
        """Create |LTISystem| from matrices stored in a .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to
            be included) containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, discretization)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            discretization which will call
            `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, discretization, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the discretization which forwards its arguments to the
            visualizer's `visualize` method.
        cache_region
            `None` or name of the cache region to use. See
            :mod:`pymor.core.cache`.
        name
            Name of the system.

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

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time,
                                 solver_options=solver_options, estimator=estimator, visualizer=visualizer,
                                 cache_region=cache_region, name=name)

    @classmethod
    def from_abcde_files(cls, files_basename, cont_time=True,
                         solver_options=None, estimator=None, visualizer=None,
                         cache_region='memory', name=None):
        """Create |LTISystem| from matrices stored in a .[ABCDE] files.

        Parameters
        ----------
        files_basename
            The basename of files containing A, B, C, and optionally D
            and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, discretization)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            discretization which will call
            `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, discretization, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the discretization which forwards its arguments to the
            visualizer's `visualize` method.
        cache_region
            `None` or name of the cache region to use. See
            :mod:`pymor.core.cache`.
        name
            Name of the system.

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

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time,
                                 solver_options=solver_options, estimator=estimator, visualizer=visualizer,
                                 cache_region=cache_region, name=name)

    def __mul__(self, other):
        """Multiply (cascade) two |LTISystems|."""
        assert self.B.source == other.C.range

        A = BlockOperator([[self.A, Concatenation(self.B, other.C)],
                           [None, other.A]])
        B = BlockColumnOperator([Concatenation(self.B, other.D), other.B])
        C = BlockRowOperator([self.C, Concatenation(self.D, other.C)])
        D = Concatenation(self.D, other.D)
        E = BlockDiagonalOperator((self.E, other.E))

        return self.__class__(A, B, C, D, E, cont_time=self.cont_time)

    @cached
    def poles(self):
        """Compute system poles."""
        if not (isinstance(self.A, NumpyMatrixOperator) and not self.A.sparse):
            self.logger.warn('Converting operator A to a NumPy array.')
        A = to_matrix(self.A, format='dense')

        if not ((isinstance(self.E, NumpyMatrixOperator) and not self.E.sparse) or isinstance(self.E, IdentityOperator)):
            self.logger.warn('Converting operator E to a NumPy array.')
        E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E, format='dense')

        return spla.eigvals(A, E)

    def eval_tf(self, s):
        """Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C (s E - A)^{-1} B + D.

        .. note::
            We assume that either the number of inputs or the number of
            outputs is much smaller than the order of the system.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        sEmA = s * E - A
        if self.m <= self.p:
            tfs = C.apply(sEmA.apply_inverse(B.as_range_array())).to_numpy().T
        else:
            tfs = B.apply_adjoint(sEmA.apply_inverse_adjoint(C.as_source_array())).to_numpy().conj()
        if not isinstance(D, ZeroOperator):
            tfs += to_matrix(D, format='dense')
        return tfs

    def eval_dtf(self, s):
        """Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C (s E - A)^{-1} E (s E - A)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of
            outputs is much smaller than the order of the system.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex
            number `s`, |NumPy array| of shape `(self.p, self.m)`.
        """
        A = self.A
        B = self.B
        C = self.C
        E = self.E

        sEmA = (s * E - A).assemble()
        if self.m <= self.p:
            dtfs = -C.apply(sEmA.apply_inverse(E.apply(sEmA.apply_inverse(B.as_range_array())))).to_numpy().T
        else:
            dtfs = -B.apply_adjoint(sEmA.apply_inverse_adjoint(E.apply_adjoint(sEmA.apply_inverse_adjoint(
                C.as_source_array())))).to_numpy().conj()
        return dtfs

    @defaults('default_solver_backend')
    def _lyap_solver(self, default_solver_backend=_DEFAULT_ME_SOLVER_BACKEND):
        options = self.solver_options.get('lyap') if self.solver_options else None
        if options:
            solver = options if isinstance(options, str) else options['type']
            backend = solver.split('_')[0]
        else:
            backend = default_solver_backend
        if backend == 'scipy':
            from pymor.bindings.scipy import solve_lyap as solve_lyap_impl
        elif backend == 'slycot':
            from pymor.bindings.slycot import solve_lyap as solve_lyap_impl
        elif backend == 'pymess':
            from pymor.bindings.pymess import solve_lyap as solve_lyap_impl
        elif backend == 'lradi':
            from pymor.algorithms.lyapunov import solve_lyap as solve_lyap_impl
        else:
            raise NotImplementedError
        return partial(solve_lyap_impl, options=options)

    @cached
    def gramian(self, typ):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'cf'`: controllability Gramian factor,
            - `'of'`: observability Gramian factor.

        Returns
        -------
        Gramian factor as a |VectorArray| from `self.A.source`.
        """
        assert isinstance(typ, str)

        if not self.cont_time:
            raise NotImplementedError

        A = self.A
        B = self.B
        C = self.C
        E = self.E if not isinstance(self.E, IdentityOperator) else None

        if typ == 'cf':
            return self._lyap_solver()(A, E, B, trans=False)
        elif typ == 'of':
            return self._lyap_solver()(A, E, C, trans=True)
        else:
            raise NotImplementedError("Only 'cf' and 'of' types are possible.")

    @cached
    def _hsv_U_V(self):
        """Compute Hankel singular values and vectors.

        Returns
        -------
        hsv
            One-dimensional |NumPy array| of singular values.
        Uh
            |NumPy array| of left singluar vectors.
        Vh
            |NumPy array| of right singluar vectors.
        """
        cf = self.gramian('cf')
        of = self.gramian('of')

        U, hsv, Vh = spla.svd(self.E.apply2(of, cf), lapack_driver='gesvd')
        return hsv, U.T, Vh

    def hsv(self):
        """Hankel singular values.

        Returns
        -------
        sv
            One-dimensional |NumPy array| of singular values.
        """
        return self._hsv_U_V()[0]

    def hsU(self):
        """Left Hankel singular vectors.

        Returns
        -------
        Uh
            |NumPy array| of left singluar vectors.
        """
        return self._hsv_U_V()[1]

    def hsV(self):
        """Right Hankel singular vectors.

        Returns
        -------
        Vh
            |NumPy array| of right singluar vectors.
        """
        return self._hsv_U_V()[2]

    @cached
    def h2_norm(self):
        """Compute the H2-norm of the |LTISystem|."""
        if not self.cont_time:
            raise NotImplementedError
        if self.m <= self.p:
            cf = self.gramian('cf')
            return np.sqrt(self.C.apply(cf).l2_norm2().sum())
        else:
            of = self.gramian('of')
            return np.sqrt(self.B.apply_adjoint(of).l2_norm2().sum())

    @cached
    def hinf_norm(self, return_fpeak=False, ab13dd_equilibrate=False):
        """Compute the H_infinity-norm of the |LTISystem|.

        Parameters
        ----------
        return_fpeak
            Should the frequency at which the maximum is achieved should
            be returned.
        ab13dd_equilibrate
            Should `slycot.ab13dd` use equilibration.

        Returns
        -------
        norm
            H_infinity-norm.
        fpeak
            Frequency at which the maximum is achieved.
        """
        for op_name in ['A', 'B', 'C']:
            if not (isinstance(getattr(self, op_name), NumpyMatrixOperator) and not getattr(self, op_name).sparse):
                self.logger.warn('Converting operator ' + op_name + ' to a NumPy array.')
        if not ((isinstance(self.D, NumpyMatrixOperator) and not self.D.sparse) or isinstance(self.D, ZeroOperator)):
            self.logger.warn('Converting operator D to a NumPy array.')
        if not ((isinstance(self.E, NumpyMatrixOperator) and not self.E.sparse) or isinstance(self.E, IdentityOperator)):
            self.logger.warn('Converting operator E to a NumPy array.')

        from slycot import ab13dd
        dico = 'C' if self.cont_time else 'D'
        jobe = 'I' if isinstance(self.E, IdentityOperator) else 'G'
        equil = 'S' if ab13dd_equilibrate else 'N'
        jobd = 'Z' if isinstance(self.D, ZeroOperator) else 'D'
        A, B, C, D, E = map(lambda op: to_matrix(op, format='dense'),
                            (self.A, self.B, self.C, self.D, self.E))
        norm, fpeak = ab13dd(dico, jobe, equil, jobd, self.n, self.m, self.p, A, E, B, C, D)

        if return_fpeak:
            return norm, fpeak
        else:
            return norm

    def hankel_norm(self):
        """Compute the Hankel-norm of the |LTISystem|."""
        return self.hsv()[0]


class TransferFunction(InputOutputSystem):
    """Class for systems represented by a transfer function.

    This class describes input-output systems given by a transfer
    function :math:`H(s)`.

    Parameters
    ----------
    input_space
        The input |VectorSpace|. Typically `NumpyVectorSpace(m, 'INPUT')` where
        m is the number of inputs.
    output_space
        The output |VectorSapce|. Typically `NumpyVectorSpace(p, 'OUTPUT')` where
        p is the number of outputs.
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

    def __init__(self, input_space, output_space, H, dH, cont_time=True, cache_region='memory', name=None):
        assert cont_time in (True, False)

        self.tf = H
        self.dtf = dH
        super().__init__(input_space, output_space, cont_time=cont_time, cache_region=cache_region, name=name)

    def eval_tf(self, s):
        return self.tf(s)

    def eval_dtf(self, s):
        return self.dtf(s)


class SecondOrderSystem(InputStateOutputSystem):
    r"""Class for linear second order systems.

    This class describes input-output systems given by

    .. math::
        M x''(t) + E x'(t) + K x(t) &= B u(t) \\
                               y(t) &= C_p x(t) + C_v x'(t) + D u(t)

    if continuous-time, or

    .. math::
        M x(k + 2) + E x(k + 1) + K x(k) &= B u(k) \\
                                    y(k) &= C_p x(k) + C_v x(k + 1) + D u(k)

    if discrete-time, where :math:`M`, :math:`E`, :math:`K`, :math:`B`,
    :math:`C_p`, :math:`C_v`, and :math:`D` are linear operators.

    Parameters
    ----------
    M
        The |Operator| M.
    E
        The |Operator| E.
    K
        The |Operator| K.
    B
        The |Operator| B.
    Cp
        The |Operator| Cp.
    Cv
        The |Operator| Cv or `None` (then Cv is assumed to be zero).
    D
        The |Operator| D or `None` (then D is assumed to be zero).
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    solver_options
        The solver options to use to solve the Lyapunov equations.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call
        `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
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
    E
        The |Operator| E.
    K
        The |Operator| K.
    B
        The |Operator| B.
    Cp
        The |Operator| Cp.
    Cv
        The |Operator| Cv.
    D
        The |Operator| D.
    operators
        Dictionary of all |Operators| contained in the discretization.
    """

    special_operators = frozenset({'M', 'E', 'K', 'B', 'Cp', 'Cv', 'D'})

    def __init__(self, M, E, K, B, Cp, Cv=None, D=None, cont_time=True,
                 solver_options=None, estimator=None, visualizer=None,
                 cache_region='memory', name=None):

        Cv = Cv or ZeroOperator(Cp.range, Cp.source)
        D = D or ZeroOperator(Cp.range, B.source)

        assert M.linear and M.source == M.range
        assert E.linear and E.source == E.range == M.source
        assert K.linear and K.source == K.range == M.source
        assert B.linear and B.range == M.source
        assert Cp.linear and Cp.source == M.range
        assert Cv.linear and Cv.source == M.range and Cv.range == Cp.range
        assert D.linear and D.source == B.source and D.range == Cp.range
        assert cont_time in (True, False)

        super().__init__(B.source, M.source, Cp.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name,
                         M=M, E=E, K=K, B=B, Cp=Cp, Cv=Cv, D=D)

        self.solution_space = M.source
        self.solver_options = solver_options

    @classmethod
    def from_matrices(cls, M, E, K, B, Cp, Cv=None, D=None, cont_time=True,
                      input_id='INPUT', state_id='STATE', output_id='OUTPUT',
                      solver_options=None, estimator=None, visualizer=None,
                      cache_region='memory', name=None):
        """Create a second order system from matrices.

        Parameters
        ----------
        M
            The |NumPy array| or |SciPy spmatrix| M.
        E
            The |NumPy array| or |SciPy spmatrix| E.
        K
            The |NumPy array| or |SciPy spmatrix| K.
        B
            The |NumPy array| or |SciPy spmatrix| B.
        Cp
            The |NumPy array| or |SciPy spmatrix| Cp.
        Cv
            The |NumPy array| or |SciPy spmatrix| Cv or `None` (then Cv
            is assumed to be zero).
        D
            The |NumPy array| or |SciPy spmatrix| D or `None` (then D
            is assumed to be zero).
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, discretization)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            discretization which will call
            `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, discretization, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the discretization which forwards its arguments to the
            visualizer's `visualize` method.
        cache_region
            `None` or name of the cache region to use. See
            :mod:`pymor.core.cache`.
        name
            Name of the system.

        Returns
        -------
        lti
            The SecondOrderSystem with operators M, E, K, B, Cp, Cv, and D.
        """
        assert isinstance(M, (np.ndarray, sps.spmatrix))
        assert isinstance(E, (np.ndarray, sps.spmatrix))
        assert isinstance(K, (np.ndarray, sps.spmatrix))
        assert isinstance(B, (np.ndarray, sps.spmatrix))
        assert isinstance(Cp, (np.ndarray, sps.spmatrix))
        assert Cv is None or isinstance(Cv, (np.ndarray, sps.spmatrix))
        assert D is None or isinstance(D, (np.ndarray, sps.spmatrix))

        M = NumpyMatrixOperator(M, source_id=state_id, range_id=state_id)
        E = NumpyMatrixOperator(E, source_id=state_id, range_id=state_id)
        K = NumpyMatrixOperator(K, source_id=state_id, range_id=state_id)
        B = NumpyMatrixOperator(B, source_id=input_id, range_id=state_id)
        Cp = NumpyMatrixOperator(Cp, source_id=state_id, range_id=output_id)
        if Cv is not None:
            Cv = NumpyMatrixOperator(Cv, source_id=state_id, range_id=output_id)
        if D is not None:
            D = NumpyMatrixOperator(D, source_id=input_id, range_id=output_id)

        return cls(M, E, K, B, Cp, Cv, D, cont_time=cont_time,
                   solver_options=solver_options, estimator=estimator, visualizer=visualizer,
                   cache_region=cache_region, name=name)

    @cached
    def to_lti(self):
        r"""Return a first order representation.

        The first order representation

        .. math::
            \begin{bmatrix}
                I & 0 \\
                0 & M
            \end{bmatrix}
            x'(t) & =
            \begin{bmatrix}
                0 & I \\
                -K & -E
            \end{bmatrix}
            x(t) +
            \begin{bmatrix}
                0 \\
                B
            \end{bmatrix}
            u(t) \\
            y(t) & =
            \begin{bmatrix}
                C_p & C_v
            \end{bmatrix}
            x(t) + D u(t)

        is returned.

        Returns
        -------
        lti
            |LTISystem| equivalent to the second order system.
        """
        state_id = self.state_space.id
        return LTISystem(A=SecondOrderSystemOperator(self.E, self.K),
                         B=BlockColumnOperator([ZeroOperator(self.B.range, self.B.source), self.B], range_id=state_id),
                         C=BlockRowOperator([self.Cp, self.Cv], source_id=state_id),
                         D=self.D,
                         E=(IdentityOperator(BlockVectorSpace([self.M.source, self.M.source], state_id)) if isinstance(self.M, IdentityOperator) else
                            BlockDiagonalOperator([IdentityOperator(self.M.source), self.M],
                                                  source_id=state_id, range_id=state_id)),
                         cont_time=self.cont_time,
                         solver_options=self.solver_options, estimator=self.estimator, visualizer=self.visualizer,
                         cache_region=self.cache_region, name=self.name + '_first_order')

    @cached
    def poles(self):
        """Compute system poles."""
        return self.to_lti().poles()

    def eval_tf(self, s):
        """Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            (C_p + s C_v) (s^2 M + s E + K)^{-1} B + D.

        .. note::
            We assume that either the number of inputs or the number of
            outputs is much smaller than the order of the system.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        M = self.M
        E = self.E
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv
        D = self.D

        s2MpsEpK = s**2 * M + s * E + K
        if self.m <= self.p:
            CppsCv = Cp + s * Cv
            tfs = CppsCv.apply(s2MpsEpK.apply_inverse(B.as_range_array())).to_numpy().T
        else:
            tfs = B.apply_adjoint(s2MpsEpK.apply_inverse_adjoint(
                Cp.as_source_array() + Cv.as_source_array() * s.conjugate())).to_numpy().conj()
        if isinstance(D, ZeroOperator):
            tfs += to_matrix(D, format='dense')
        return tfs

    def eval_dtf(self, s):
        """Evaluate the derivative of the transfer function.

        .. math::
            s C_v (s^2 M + s E + K)^{-1} B
            - (C_p + s C_v) (s^2 M + s E + K)^{-1} (2 s M + E)
                (s^2 M + s E + K)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of
            outputs is much smaller than the order of the system.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex
            number `s`, |NumPy array| of shape `(self.p, self.m)`.
        """
        M = self.M
        E = self.E
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv

        s2MpsEpK = (s**2 * M + s * E + K).assemble()
        sM2pE = 2 * s * M + E
        if self.m <= self.p:
            dtfs = Cv.apply(s2MpsEpK.apply_inverse(B.as_range_array())).to_numpy().T * s
            CppsCv = Cp + s * Cv
            dtfs -= CppsCv.apply(s2MpsEpK.apply_inverse(sM2pE.apply(s2MpsEpK.apply_inverse(
                B.as_range_array())))).to_numpy().T
        else:
            dtfs = B.apply_adjoint(s2MpsEpK.apply_inverse_adjoint(Cv.as_source_array())).to_numpy().conj() * s
            dtfs -= B.apply_adjoint(s2MpsEpK.apply_inverse_adjoint(sM2pE.apply_adjoint(
                s2MpsEpK.apply_inverse_adjoint(Cp.as_source_array() +
                                               Cv.as_source_array() * s.conjugate())))).to_numpy().conj()
        return dtfs

    @cached
    def gramian(self, typ):
        """Compute a second-order Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'pcf'`: position controllability Gramian factor,
            - `'vcf'`: velocity controllability Gramian factor,
            - `'pof'`: position observability Gramian factor,
            - `'vof'`: velocity observability Gramian factor.

        Returns
        -------
        Gramian factor as a |VectorArray| from `self.M.source`.
        """
        if typ == 'pcf':
            return self.to_lti().gramian('cf').block(0)
        elif typ == 'vcf':
            return self.to_lti().gramian('cf').block(1)
        elif typ == 'pof':
            return self.to_lti().gramian('of').block(0)
        elif typ == 'vof':
            return self.to_lti().gramian('of').block(1)
        else:
            raise NotImplementedError("Only 'pcf', 'vcf', 'pof', and 'vof' types are possible.")

    def psv(self):
        """Position singular values.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(self.gramian('pof').inner(self.gramian('pcf')))

    def vsv(self):
        """Velocity singular values.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(self.gramian('vof').inner(self.gramian('vcf'), product=self.M))

    def pvsv(self):
        """Position-velocity singular values.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(self.gramian('vof').inner(self.gramian('pcf'), product=self.M))

    def vpsv(self):
        """Velocity-position singular values.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(self.gramian('pof').inner(self.gramian('vcf')))

    @cached
    def h2_norm(self):
        """Compute the H2-norm."""
        return self.to_lti().h2_norm()

    @cached
    def hinf_norm(self, return_fpeak=False, ab13dd_equilibrate=False):
        """Compute the H_infinity-norm.

        Parameters
        ----------
        return_fpeak
            Should the frequency at which the maximum is achieved should
            be returned.
        ab13dd_equilibrate
            Should `slycot.ab13dd` use equilibration.

        Returns
        -------
        norm
            H_infinity-norm.
        fpeak
            Frequency at which the maximum is achieved.
        """
        return self.to_lti().hinf_norm(return_fpeak=return_fpeak,
                                       ab13dd_equilibrate=ab13dd_equilibrate)

    @cached
    def hankel_norm(self):
        """Compute the Hankel-norm."""
        return self.to_lti().hankel_norm()


class LinearDelaySystem(InputStateOutputSystem):
    r"""Class for linear delay systems.

    This class describes input-state-output systems given by

    .. math::
        E x'(t)
        & = A x(t) + \sum_{i = 1}^q{A_i x(t - \tau_i)} + B u(t) \\
        y(t)
        & = C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1)
        & = A x(k) + \sum_{i = 1}^q{A_i x(k - \tau_i)} + B u(k) \\
        y(k)
        &= C x(k) + D u(k)

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`,
    :math:`B`, :math:`C`, and :math:`D` are linear operators.

    Parameters
    ----------
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
    D
        The |Operator| D or `None` (then D is assumed to be zero).
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call
        `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
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
    tau
        The tuple of delay times.
    A
        The |Operator| A.
    Ad
        The tuple of |Operators| A_i.
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

    special_operators = frozenset({'A', 'Ad', 'B', 'C', 'D', 'E'})

    def __init__(self, A, Ad, tau, B, C, D=None, E=None, cont_time=True,
                 estimator=None, visualizer=None,
                 cache_region='memory', name=None):

        D = D or ZeroOperator(C.range, B.source)
        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range
        assert isinstance(Ad, tuple) and len(Ad) > 0
        assert all(Ai.linear and Ai.source == Ai.range == A.source for Ai in Ad)
        assert isinstance(tau, tuple) and len(tau) == len(Ad) and all(taui > 0 for taui in tau)
        assert B.linear and B.range == A.source
        assert C.linear and C.source == A.range
        assert D.linear and D.source == B.source and D.range == C.range
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name,
                         A=A, Ad=Ad, B=B, C=C, D=D, E=E)

        self.solution_space = A.source
        self.tau = tau
        self.q = len(Ad)

    def eval_tf(self, s):
        r"""Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C \left(s E - A
                - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B + D.

        .. note::
            We assume that either the number of inputs or the number of
            outputs is much smaller than the order of the system.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`,
            |NumPy array| of shape `(self.p, self.m)`.
        """
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        middle = LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        if self.m <= self.p:
            tfs = C.apply(middle.apply_inverse(B.as_range_array())).to_numpy().T
        else:
            tfs = B.apply_adjoint(middle.apply_inverse_adjoint(C.as_source_array())).to_numpy().conj()
        if isinstance(D, ZeroOperator):
            tfs += to_matrix(D, format='dense')
        return tfs

    def eval_dtf(self, s):
        r"""Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C \left(s E - A
                    - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1}
                \left(E
                    + \sum_{i = 1}^q{\tau_i e^{-\tau_i s} A_i}\right)
                \left(s E - A
                    - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B.

        .. note::
            We assume that either the number of inputs or the number of
            outputs is much smaller than the order of the system.

        Parameters
        ----------
        s
            Complex number.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex
            number `s`, |NumPy array| of shape `(self.p, self.m)`.
        """
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C
        E = self.E

        left_and_right = LincombOperator((E, A) + Ad,
                                         (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau)).assemble()
        middle = LincombOperator((E,) + Ad, (1,) + tuple(taui * np.exp(-taui * s) for taui in self.tau))
        if self.m <= self.p:
            dtfs = -C.apply(left_and_right.apply_inverse(middle.apply(left_and_right.apply_inverse(
                B.as_range_array())))).to_numpy().T
        else:
            dtfs = -B.apply_adjoint(left_and_right.apply_inverse_adjoint(middle.apply_adjoint(
                left_and_right.apply_inverse_adjoint(C.as_source_array())))).to_numpy().conj()
        return dtfs


class LinearStochasticSystem(InputStateOutputSystem):
    r"""Class for linear stochastic systems.

    This class describes input-state-output systems given by

    .. math::
        E \mathrm{d}x(t)
        & =
            A x(t) \mathrm{d}t
            + \sum_{i = 1}^q{A_i x(t) \mathrm{d}\omega_i(t)}
            + B u(t) \mathrm{d}t \\
        y(t)
        & = C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1)
        & =
            A x(k)
            + \sum_{i = 1}^q{A_i x(k) \omega_i(k)}
            + B u(k) \\
        y(k)
        & = C x(k) + D u(t)

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`,
    :math:`B`, :math:`C`, and :math:`D` are linear operators and
    :math:`\omega_i` are stochastic processes.

    Parameters
    ----------
    A
        The |Operator| A.
    As
        The tuple of |Operators| A_i.
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
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call
        `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
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
    A
        The |Operator| A.
    As
        The tuple of |Operators| A_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    D
        The |Operator| D.
    E
        The |Operator| E.
    operators
        Dictionary of all |Operators| contained in the discretization.
    """

    special_operators = frozenset({'A', 'As', 'B', 'C', 'D', 'E'})

    def __init__(self, A, As, B, C, D=None, E=None, cont_time=True,
                 estimator=None, visualizer=None,
                 cache_region='memory', name=None):

        D = D or ZeroOperator(C.range, B.source)
        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range
        assert isinstance(As, tuple) and len(As) > 0
        assert all(Ai.linear and Ai.source == Ai.range == A.source for Ai in As)
        assert B.linear and B.range == A.source
        assert C.linear and C.source == A.range
        assert D.linear and D.source == B.source and D.range == C.range
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name,
                         A=A, As=As, B=B, C=C, D=D, E=E)

        self.solution_space = A.source
        self.q = len(As)


class BilinearSystem(InputStateOutputSystem):
    r"""Class for bilinear systems.

    This class describes input-output systems given by

    .. math::
        E x'(t)
        & =
            A x(t)
            + \sum_{i = 1}^m{N_i x(t) u_i(t)}
            + B u(t) \\
        y(t)
        & = C x(t) + D u(t)

    if continuous-time, or

    .. math::
        E x(k + 1)
        & =
            A x(k)
            + \sum_{i = 1}^m{N_i x(k) u_i(k)}
            + B u(k) \\
        y(k)
        & = C x(k) + D u(t)

    if discrete-time, where :math:`E`, :math:`A`, :math:`N_i`,
    :math:`B`, :math:`C`, and :math:`D` are linear operators and
    :math:`m` is the number of inputs.

    Parameters
    ----------
    A
        The |Operator| A.
    N
        The tuple of |Operators| N_i.
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
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, discretization)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        discretization which will call
        `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, discretization, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the discretization which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the cache region to use. See
        :mod:`pymor.core.cache`.
    name
        Name of the system.

    Attributes
    ----------
    n
        The order of the system (equal to A.source.dim).
    A
        The |Operator| A.
    N
        The tuple of |Operators| N_i.
    B
        The |Operator| B.
    C
        The |Operator| C.
    D
        The |Operator| D.
    E
        The |Operator| E.
    operators
        Dictionary of all |Operators| contained in the discretization.
    """

    special_operators = frozenset({'A', 'N', 'B', 'C', 'D', 'E'})

    def __init__(self, A, N, B, C, D, E=None, cont_time=True,
                 estimator=None, visualizer=None,
                 cache_region='memory', name=None):

        D = D or ZeroOperator(C.range, B.source)
        E = E or IdentityOperator(A.source)

        assert A.linear and A.source == A.range
        assert B.linear and B.range == A.source
        assert isinstance(N, tuple) and len(N) == B.source.dim
        assert all(Ni.linear and Ni.source == Ni.range == A.source for Ni in N)
        assert C.linear and C.source == A.range
        assert D.linear and D.source == B.source and D.range == C.range
        assert E.linear and E.source == E.range == A.source
        assert cont_time in (True, False)

        super().__init__(B.source, C.range, state_space=A.source, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer,
                         cache_region=cache_region, name=name,
                         A=A, N=N, B=B, C=C, D=D, E=E)

        self.solution_space = A.source
        self.linear = False
