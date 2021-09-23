# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.bernoulli import bernoulli_stabilize
from pymor.algorithms.eigs import eigs
from pymor.algorithms.lyapunov import solve_lyap_lrcf, solve_lyap_dense
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.base import abstractmethod
from pymor.core.cache import cached
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.models.interface import Model
from pymor.operators.block import (BlockOperator, BlockRowOperator, BlockColumnOperator, BlockDiagonalOperator,
                                   SecondOrderModelOperator)
from pymor.operators.constructions import IdentityOperator, LincombOperator, LowRankOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu
from pymor.vectorarrays.block import BlockVectorSpace


@defaults('value')
def sparse_min_size(value=1000):
    """Return minimal sparse problem size for which to warn about converting to dense."""
    return value


class InputOutputModel(Model):
    """Base class for input-output systems."""

    cache_region = 'memory'

    def __init__(self, dim_input, dim_output, cont_time=True,
                 error_estimator=None, visualizer=None, name=None):
        assert cont_time in (True, False)
        super().__init__(error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    @abstractmethod
    def eval_tf(self, s, mu=None):
        """Evaluate the transfer function."""

    @abstractmethod
    def eval_dtf(self, s, mu=None):
        """Evaluate the derivative of the transfer function."""

    @cached
    def freq_resp(self, w, mu=None):
        """Evaluate the transfer function on the imaginary axis.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.

        Returns
        -------
        tfw
            Transfer function values at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        """
        if not self.cont_time:
            raise NotImplementedError
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        return np.stack([self.eval_tf(1j * wi, mu=mu) for wi in w])

    def bode(self, w, mu=None):
        """Compute magnitudes and phases.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.

        Returns
        -------
        mag
            Transfer function magnitudes at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        phase
            Transfer function phases (in radians) at frequencies in `w`, |NumPy array| of shape
            `(len(w), self.dim_output, self.dim_input)`.
        """
        w = np.asarray(w)
        mag = np.abs(self.freq_resp(w, mu=mu))
        phase = np.angle(self.freq_resp(w, mu=mu))
        phase = np.unwrap(phase, axis=0)
        return mag, phase

    def bode_plot(self, w, mu=None, ax=None, Hz=False, dB=False, deg=True, **mpl_kwargs):
        """Draw the Bode plot for all input-output pairs.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter| for which to evaluate the transfer function.
        ax
            Axis of shape (2 * `self.dim_output`, `self.dim_input`) to which to plot.
            If not given, `matplotlib.pyplot.gcf` is used to get the figure and create axis.
        Hz
            Should the frequency be in Hz on the plot.
        dB
            Should the magnitude be in dB on the plot.
        deg
            Should the phase be in degrees (otherwise in radians).
        mpl_kwargs
            Keyword arguments used in the matplotlib plot function.

        Returns
        -------
        artists
            List of matplotlib artists added.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            width, height = plt.rcParams['figure.figsize']
            fig.set_size_inches(self.dim_input * width, 2 * self.dim_output * height)
            fig.set_constrained_layout(True)
            ax = fig.subplots(2 * self.dim_output, self.dim_input, sharex=True, squeeze=False)
        else:
            assert isinstance(ax, np.ndarray) and ax.shape == (2 * self.dim_output, self.dim_input)
            fig = ax[0, 0].get_figure()

        w = np.asarray(w)
        freq = w / (2 * np.pi) if Hz else w
        mag, phase = self.bode(w, mu=mu)
        if deg:
            phase *= 180 / np.pi

        artists = np.empty_like(ax)
        freq_label = f'Frequency ({"Hz" if Hz else "rad/s"})'
        mag_label = f'Magnitude{" (dB)" if dB else ""}'
        phase_label = f'Phase ({"deg" if deg else "rad"})'
        for i in range(self.dim_output):
            for j in range(self.dim_input):
                if dB:
                    artists[2 * i, j] = ax[2 * i, j].semilogx(freq, 20 * np.log10(mag[:, i, j]),
                                                              **mpl_kwargs)
                else:
                    artists[2 * i, j] = ax[2 * i, j].loglog(freq, mag[:, i, j],
                                                            **mpl_kwargs)
                artists[2 * i + 1, j] = ax[2 * i + 1, j].semilogx(freq, phase[:, i, j],
                                                                  **mpl_kwargs)
        for i in range(self.dim_output):
            ax[2 * i, 0].set_ylabel(mag_label)
            ax[2 * i + 1, 0].set_ylabel(phase_label)
        for j in range(self.dim_input):
            ax[-1, j].set_xlabel(freq_label)
        fig.suptitle('Bode plot')

        return artists

    def mag_plot(self, w, mu=None, ax=None, ord=None, Hz=False, dB=False, **mpl_kwargs):
        """Draw the magnitude plot.

        Parameters
        ----------
        w
            A sequence of angular frequencies at which to compute the transfer function.
        mu
            |Parameter values| for which to evaluate the transfer function.
        ax
            Axis to which to plot.
            If not given, `matplotlib.pyplot.gca` is used.
        ord
            The order of the norm used to compute the magnitude (the default is the Frobenius norm).
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

        w = np.asarray(w)
        freq = w / (2 * np.pi) if Hz else w
        mag = spla.norm(self.freq_resp(w, mu=mu), ord=ord, axis=(1, 2))
        if dB:
            out = ax.semilogx(freq, 20 * np.log10(mag), **mpl_kwargs)
        else:
            out = ax.loglog(freq, mag, **mpl_kwargs)

        ax.set_title('Magnitude plot')
        freq_unit = ' (Hz)' if Hz else ' (rad/s)'
        ax.set_xlabel('Frequency' + freq_unit)
        mag_unit = ' (dB)' if dB else ''
        ax.set_ylabel('Magnitude' + mag_unit)

        return out

    @cached
    def h2_norm(self, return_norm_only=True, **quad_kwargs):
        """Compute the H2-norm using quadrature.

        This method uses `scipy.integrate.quad` and makes no assumptions on the form of the transfer
        function.

        By default, the absolute error tolerance in `scipy.integrate.quad` is set to zero (see its
        optional argument `epsabs`).
        It can be changed by using the `epsabs` keyword argument.

        Parameters
        ----------
        return_norm_only
            Whether to only return the approximate H2-norm.
        quad_kwargs
            Keyword arguments passed to `scipy.integrate.quad`.

        Returns
        -------
        norm
            Computed H2-norm.
        norm_relerr
            Relative error estimate (returned if `return_norm_only` is `False`).
        info
            Quadrature info (returned if `return_norm_only` is `False` and `full_output` is `True`).
            See `scipy.integrate.quad` documentation for more details.
        """
        if not self.cont_time:
            raise NotImplementedError

        import scipy.integrate as spint
        quad_kwargs.setdefault('epsabs', 0)
        quad_out = spint.quad(lambda w: spla.norm(self.eval_tf(w * 1j))**2,
                              0, np.inf,
                              **quad_kwargs)
        norm = np.sqrt(quad_out[0] / np.pi)
        if return_norm_only:
            return norm
        norm_relerr = quad_out[1] / (2 * quad_out[0])
        if len(quad_out) == 2:
            return norm, norm_relerr
        else:
            return norm, norm_relerr, quad_out[2:]

    def h2_inner(self, lti):
        """Compute H2 inner product with an |LTIModel|.

        Uses the inner product formula based on the pole-residue form
        (see, e.g., Lemma 1 in :cite:`ABG10`).

        Parameters
        ----------
        lti
            |LTIModel| consisting of |Operators| that can be converted to |NumPy arrays|.
            The D operator is ignored.

        Returns
        -------
        inner
            H2 inner product.
        """
        assert isinstance(lti, LTIModel)

        poles, b, c = _lti_to_poles_b_c(lti)
        inner = sum(c[i].dot(self.eval_tf(-poles[i]).dot(b[i]))
                    for i in range(len(poles)))
        inner = inner.conjugate()

        return inner


class InputStateOutputModel(InputOutputModel):
    """Base class for input-output systems with state space."""

    def __init__(self, dim_input, solution_space, dim_output, cont_time=True,
                 error_estimator=None, visualizer=None, name=None):
        super().__init__(dim_input, dim_output, cont_time=cont_time,
                         error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    @property
    def order(self):
        return self.solution_space.dim


class LTIModel(InputStateOutputModel):
    r"""Class for linear time-invariant systems.

    This class describes input-state-output systems given by

    .. math::
        E(\mu) \dot{x}(t, \mu) & = A(\mu) x(t, \mu) + B(\mu) u(t), \\
                     y(t, \mu) & = C(\mu) x(t, \mu) + D(\mu) u(t),

    if continuous-time, or

    .. math::
        E(\mu) x(k + 1, \mu) & = A(\mu) x(k, \mu) + B(\mu) u(k), \\
               y(k, \mu)     & = C(\mu) x(k, \mu) + D(\mu) u(k),

    if discrete-time, where :math:`A`, :math:`B`, :math:`C`, :math:`D`, and :math:`E` are linear
    operators.

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
    error_estimator
        An error estimator for the problem. This can be any object with an
        `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
        `estimate_error(U, mu)` method is added to the model which will call
        `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with a
        `visualize(U, model, ...)` method. If `visualizer` is not `None`, a
        `visualize(U, *args, **kwargs)` method is added to the model which forwards its arguments to
        the visualizer's `visualize` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system.
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
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
    """

    def __init__(self, A, B, C, D=None, E=None, cont_time=True,
                 solver_options=None, error_estimator=None, visualizer=None, name=None):

        assert A.linear
        assert A.source == A.range
        assert B.linear
        assert B.range == A.source
        assert C.linear
        assert C.source == A.range

        D = D or ZeroOperator(C.range, B.source)
        assert D.linear
        assert D.source == B.source
        assert D.range == C.range

        E = E or IdentityOperator(A.source)
        assert E.linear
        assert E.source == E.range
        assert E.source == A.source

        assert solver_options is None or solver_options.keys() <= {'lyap_lrcf', 'lyap_dense'}

        super().__init__(B.source.dim, A.source, C.range.dim, cont_time=cont_time,
                         error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    @classmethod
    def from_matrices(cls, A, B, C, D=None, E=None, cont_time=True,
                      state_id='STATE', solver_options=None, error_estimator=None,
                      visualizer=None, name=None):
        """Create |LTIModel| from matrices.

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
            The |NumPy array| or |SciPy spmatrix| E or `None` (then E is assumed to be identity).
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        state_id
            Id of the state space.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        error_estimator
            An error estimator for the problem. This can be any object with an
            `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
            `estimate_error(U, mu)` method is added to the model which will call
            `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        lti
            The |LTIModel| with operators A, B, C, D, and E.
        """
        assert isinstance(A, (np.ndarray, sps.spmatrix))
        assert isinstance(B, (np.ndarray, sps.spmatrix))
        assert isinstance(C, (np.ndarray, sps.spmatrix))
        assert D is None or isinstance(D, (np.ndarray, sps.spmatrix))
        assert E is None or isinstance(E, (np.ndarray, sps.spmatrix))

        A = NumpyMatrixOperator(A, source_id=state_id, range_id=state_id)
        B = NumpyMatrixOperator(B, range_id=state_id)
        C = NumpyMatrixOperator(C, source_id=state_id)
        if D is not None:
            D = NumpyMatrixOperator(D)
        if E is not None:
            E = NumpyMatrixOperator(E, source_id=state_id, range_id=state_id)

        return cls(A, B, C, D, E, cont_time=cont_time,
                   solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer,
                   name=name)

    def to_matrices(self):
        """Return operators as matrices.

        Returns
        -------
        A
            The |NumPy array| or |SciPy spmatrix| A.
        B
            The |NumPy array| or |SciPy spmatrix| B.
        C
            The |NumPy array| or |SciPy spmatrix| C.
        D
            The |NumPy array| or |SciPy spmatrix| D or `None` (if D is a `ZeroOperator`).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (if E is an `IdentityOperator`).
        """
        A = to_matrix(self.A)
        B = to_matrix(self.B)
        C = to_matrix(self.C)
        D = None if isinstance(self.D, ZeroOperator) else to_matrix(self.D)
        E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E)
        return A, B, C, D, E

    @classmethod
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, cont_time=True,
                   state_id='STATE', solver_options=None, error_estimator=None, visualizer=None,
                   name=None):
        """Create |LTIModel| from matrices stored in separate files.

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
        state_id
            Id of the state space.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        error_estimator
            An error estimator for the problem. This can be any object with an
            `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
            `estimate_error(U, mu)` method is added to the model which will call
            `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        lti
            The |LTIModel| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix

        A = load_matrix(A_file)
        B = load_matrix(B_file)
        C = load_matrix(C_file)
        D = load_matrix(D_file) if D_file is not None else None
        E = load_matrix(E_file) if E_file is not None else None

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time,
                                 state_id=state_id, solver_options=solver_options,
                                 error_estimator=error_estimator, visualizer=visualizer, name=name)

    def to_files(self, A_file, B_file, C_file, D_file=None, E_file=None):
        """Write operators to files as matrices.

        Parameters
        ----------
        A_file
            The name of the file (with extension) containing A.
        B_file
            The name of the file (with extension) containing B.
        C_file
            The name of the file (with extension) containing C.
        D_file
            The name of the file (with extension) containing D or `None` if D is a `ZeroOperator`.
        E_file
            The name of the file (with extension) containing E or `None` if E is an
            `IdentityOperator`.
        """
        if D_file is None and not isinstance(self.D, ZeroOperator):
            raise ValueError('D is not zero, D_file must be given')
        if E_file is None and not isinstance(self.E, IdentityOperator):
            raise ValueError('E is not identity, E_file must be given')

        from pymor.tools.io import save_matrix

        A, B, C, D, E = self.to_matrices()
        for mat, file in [(A, A_file), (B, B_file), (C, C_file), (D, D_file), (E, E_file)]:
            if mat is None:
                continue
            save_matrix(file, mat)

    @classmethod
    def from_mat_file(cls, file_name, cont_time=True,
                      state_id='STATE', solver_options=None, error_estimator=None,
                      visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in a .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be included) containing A, B,
            C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        state_id
            Id of the state space.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        error_estimator
            An error estimator for the problem. This can be any object with an
            `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
            `estimate_error(U, mu)` method is added to the model which will call
            `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        lti
            The |LTIModel| with operators A, B, C, D, and E.
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
                                 state_id=state_id, solver_options=solver_options,
                                 error_estimator=error_estimator, visualizer=visualizer, name=name)

    def to_mat_file(self, file_name):
        """Save operators as matrices to .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be included).
        """
        import scipy.io as spio
        A, B, C, D, E = self.to_matrices()
        mat_dict = {'A': A, 'B': B, 'C': C}
        if D is not None:
            mat_dict['D'] = D
        if E is not None:
            mat_dict['E'] = E
        spio.savemat(file_name, mat_dict)

    @classmethod
    def from_abcde_files(cls, files_basename, cont_time=True,
                         state_id='STATE', solver_options=None, error_estimator=None,
                         visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in .[ABCDE] files.

        Parameters
        ----------
        files_basename
            The basename of files containing A, B, C, and optionally D and E.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        state_id
            Id of the state space.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        error_estimator
            An error estimator for the problem. This can be any object with an
            `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
            `estimate_error(U, mu)` method is added to the model which will call
            `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        lti
            The |LTIModel| with operators A, B, C, D, and E.
        """
        from pymor.tools.io import load_matrix
        import os.path

        A = load_matrix(files_basename + '.A')
        B = load_matrix(files_basename + '.B')
        C = load_matrix(files_basename + '.C')
        D = load_matrix(files_basename + '.D') if os.path.isfile(files_basename + '.D') else None
        E = load_matrix(files_basename + '.E') if os.path.isfile(files_basename + '.E') else None

        return cls.from_matrices(A, B, C, D, E, cont_time=cont_time,
                                 state_id=state_id, solver_options=solver_options,
                                 error_estimator=error_estimator, visualizer=visualizer, name=name)

    def to_abcde_files(self, files_basename):
        """Save operators as matrices to .[ABCDE] files in Matrix Market format.

        Parameters
        ----------
        files_basename
            The basename of files containing the operators.
        """
        from pathlib import Path
        from pymor.tools.io.matrices import _mmwrite
        A, B, C, D, E = self.to_matrices()
        _mmwrite(Path(files_basename + '.A'), A)
        _mmwrite(Path(files_basename + '.B'), B)
        _mmwrite(Path(files_basename + '.C'), C)
        if D is not None:
            _mmwrite(Path(files_basename + '.D'), D)
        if E is not None:
            _mmwrite(Path(files_basename + '.E'), E)

    def __add__(self, other):
        """Add an |LTIModel|."""
        if not isinstance(other, LTIModel):
            return NotImplemented

        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.source
        assert self.D.range == other.D.range

        A = BlockDiagonalOperator([self.A, other.A])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, other.C])
        D = self.D + other.D
        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            E = BlockDiagonalOperator([self.E, other.E])
        return self.with_(A=A, B=B, C=C, D=D, E=E)

    def __sub__(self, other):
        """Subtract an |LTIModel|."""
        return self + (-other)

    def __neg__(self):
        """Negate the |LTIModel|."""
        return self.with_(C=-self.C, D=-self.D)

    def __mul__(self, other):
        """Postmultiply by an |LTIModel|."""
        if not isinstance(other, LTIModel):
            return NotImplemented

        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.range

        A = BlockOperator([[self.A, self.B @ other.C],
                           [None, other.A]])
        B = BlockColumnOperator([self.B @ other.D, other.B])
        C = BlockRowOperator([self.C, self.D @ other.C])
        D = self.D @ other.D
        E = BlockDiagonalOperator([self.E, other.E])
        return self.with_(A=A, B=B, C=C, D=D, E=E)

    @cached
    def poles(self, mu=None):
        """Compute system poles.

        .. note::
            Assumes the systems is small enough to use a dense eigenvalue solver.

        Parameters
        ----------
        mu
            |Parameter values| for which to compute the systems poles.

        Returns
        -------
        One-dimensional |NumPy array| of system poles.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A.assemble(mu=mu)
        E = self.E.assemble(mu=mu)

        if self.order >= sparse_min_size():
            if not isinstance(A, NumpyMatrixOperator) or A.sparse:
                self.logger.warning('Converting operator A to a NumPy array.')
            if not isinstance(E, IdentityOperator):
                if not isinstance(E, NumpyMatrixOperator) or E.sparse:
                    self.logger.warning('Converting operator E to a NumPy array.')

        A = to_matrix(A, format='dense')
        E = None if isinstance(E, IdentityOperator) else to_matrix(E, format='dense')
        return spla.eigvals(A, E)

    def eval_tf(self, s, mu=None):
        r"""Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C(\mu) (s E(\mu) - A(\mu))^{-1} B(\mu) + D(\mu).

        .. note::
            Assumes that the number of inputs and outputs is much smaller than the order of the
            system.

        Parameters
        ----------
        s
            Complex number.
        mu
            |Parameter values|.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array| of shape
            `(self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        sEmA = s * E - A
        if self.dim_input <= self.dim_output:
            tfs = C.apply(sEmA.apply_inverse(B.as_range_array(mu=mu),
                                             mu=mu),
                          mu=mu).to_numpy().T
        else:
            tfs = B.apply_adjoint(sEmA.apply_inverse_adjoint(C.as_source_array(mu=mu),
                                                             mu=mu),
                                  mu=mu).to_numpy().conj()
        if not isinstance(D, ZeroOperator):
            tfs += to_matrix(D, format='dense', mu=mu)
        return tfs

    def eval_dtf(self, s, mu=None):
        r"""Evaluate the derivative of the transfer function.

        The derivative of the transfer function at :math:`s` is

        .. math::
            -C(\mu) (s E(\mu) - A(\mu))^{-1} E(\mu)
                (s E(\mu) - A(\mu))^{-1} B(\mu).

        .. note::
            Assumes that the number of inputs and outputs is much smaller than the order of the
            system.

        Parameters
        ----------
        s
            Complex number.
        mu
            |Parameter values|.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`, |NumPy array| of
            shape `(self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A
        B = self.B
        C = self.C
        E = self.E

        sEmA = (s * E - A).assemble(mu=mu)
        if self.dim_input <= self.dim_output:
            dtfs = -C.apply(
                sEmA.apply_inverse(
                    E.apply(
                        sEmA.apply_inverse(
                            B.as_range_array(mu=mu)),
                        mu=mu)),
                mu=mu).to_numpy().T
        else:
            dtfs = -B.apply_adjoint(
                sEmA.apply_inverse_adjoint(
                    E.apply_adjoint(
                        sEmA.apply_inverse_adjoint(
                            C.as_source_array(mu=mu)),
                        mu=mu)),
                mu=mu).to_numpy().conj()
        return dtfs

    @cached
    def gramian(self, typ, mu=None):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'c_lrcf'`: low-rank Cholesky factor of the controllability Gramian,
            - `'o_lrcf'`: low-rank Cholesky factor of the observability Gramian,
            - `'c_dense'`: dense controllability Gramian,
            - `'o_dense'`: dense observability Gramian.

            .. note::
                For `'c_lrcf'` and `'o_lrcf'` types, the method assumes the system is asymptotically
                stable.
                For `'c_dense'` and `'o_dense'` types, the method assumes there are no two system
                poles which add to zero.
        mu
            |Parameter values|.

        Returns
        -------
        If typ is `'c_lrcf'` or `'o_lrcf'`, then the Gramian factor as a |VectorArray| from
        `self.A.source`.
        If typ is `'c_dense'` or `'o_dense'`, then the Gramian as a |NumPy array|.
        """
        if not self.cont_time:
            raise NotImplementedError

        assert typ in ('c_lrcf', 'o_lrcf', 'c_dense', 'o_dense')

        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A.assemble(mu)
        B = self.B
        C = self.C
        E = self.E.assemble(mu) if not isinstance(self.E, IdentityOperator) else None
        options_lrcf = self.solver_options.get('lyap_lrcf') if self.solver_options else None
        options_dense = self.solver_options.get('lyap_dense') if self.solver_options else None

        if typ == 'c_lrcf':
            return solve_lyap_lrcf(A, E, B.as_range_array(mu=mu),
                                   trans=False, options=options_lrcf)
        elif typ == 'o_lrcf':
            return solve_lyap_lrcf(A, E, C.as_source_array(mu=mu),
                                   trans=True, options=options_lrcf)
        elif typ == 'c_dense':
            return solve_lyap_dense(to_matrix(A, format='dense'),
                                    to_matrix(E, format='dense') if E else None,
                                    to_matrix(B, format='dense'),
                                    trans=False, options=options_dense)
        elif typ == 'o_dense':
            return solve_lyap_dense(to_matrix(A, format='dense'),
                                    to_matrix(E, format='dense') if E else None,
                                    to_matrix(C, format='dense'),
                                    trans=True, options=options_dense)

    @cached
    def _hsv_U_V(self, mu=None):
        """Compute Hankel singular values and vectors.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        hsv
            One-dimensional |NumPy array| of singular values.
        Uh
            |NumPy array| of left singular vectors.
        Vh
            |NumPy array| of right singular vectors.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        cf = self.gramian('c_lrcf', mu=mu)
        of = self.gramian('o_lrcf', mu=mu)
        U, hsv, Vh = spla.svd(self.E.apply2(of, cf, mu=mu), lapack_driver='gesvd')
        return hsv, U.T, Vh

    def hsv(self, mu=None):
        """Hankel singular values.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        sv
            One-dimensional |NumPy array| of singular values.
        """
        return self._hsv_U_V(mu=mu)[0]

    @cached
    def h2_norm(self, mu=None):
        """Compute the H2-norm of the |LTIModel|.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        norm
            H_2-norm.
        """
        if not self.cont_time:
            raise NotImplementedError
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        D_norm2 = np.sum(self.D.as_range_array(mu=mu).norm2())
        if D_norm2 != 0:
            self.logger.warning('The D operator is not exactly zero '
                                f'(squared Frobenius norm is {D_norm2}).')
        assert self.parameters.assert_compatible(mu)
        if self.dim_input <= self.dim_output:
            cf = self.gramian('c_lrcf', mu=mu)
            return np.sqrt(self.C.apply(cf, mu=mu).norm2().sum())
        else:
            of = self.gramian('o_lrcf', mu=mu)
            return np.sqrt(self.B.apply_adjoint(of, mu=mu).norm2().sum())

    @cached
    def hinf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False):
        """Compute the H_infinity-norm of the |LTIModel|.

        .. note::
            Assumes the system is asymptotically stable. Under this is assumption
            the H_infinity-norm is equal to the L_infinity-norm. Accordingly, this
            method calls :meth:`~pymor.models.iosys.LTIModel.linf_norm`.

        Parameters
        ----------
        mu
            |Parameter values|.
        return_fpeak
            Whether to return the frequency at which the maximum is achieved.
        ab13dd_equilibrate
            Whether `slycot.ab13dd` should use equilibration.

        Returns
        -------
        norm
            H_infinity-norm.
        fpeak
            Frequency at which the maximum is achieved (if `return_fpeak` is `True`).
        """
        return self.linf_norm(mu=mu, return_fpeak=return_fpeak, ab13dd_equilibrate=ab13dd_equilibrate)

    def hankel_norm(self, mu=None):
        """Compute the Hankel-norm of the |LTIModel|.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        norm
            Hankel-norm.
        """
        return self.hsv(mu=mu)[0]

    def l2_norm(self, ast_pole_data=None, mu=None):
        r"""Compute the L2-norm of the |LTIModel|.

        The L2-norm of an |LTIModel| is defined via the integral

        .. math::
            \lVert H \rVert_{\mathcal{L}_2}
            =
            \left(
              \frac{1}{2 \pi}
              \int_{-\infty}^{\infty}
              \lVert H(\boldsymbol{\imath} \omega) \rVert_{\operatorname{F}}^2
              \operatorname{d}\!\omega
            \right)^{\frac{1}{2}}.

        Parameters
        ----------
        ast_pole_data
            Can be:

            - dictionary of parameters for :func:`~pymor.algorithms.eigs.eigs`,
            - list of anti-stable eigenvalues (scalars),
            - tuple `(lev, ew, rev)` where `ew` contains the anti-stable eigenvalues
              and `lev` and `rev` are |VectorArrays| representing the eigenvectors.
            - `None` if anti-stable eigenvalues should be computed via dense methods.
        mu
            |Parameter|.

        Returns
        -------
        norm
            L_2-norm.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        A, B, C, D, E = (op.assemble(mu=mu) for op in [self.A, self.B, self.C, self.D, self.E])
        options_lrcf = self.solver_options.get('lyap_lrcf') if self.solver_options else None

        ast_spectrum = self.get_ast_spectrum(ast_pole_data, mu)

        if len(ast_spectrum[0]) == 0:
            return self.h2_norm()

        K = bernoulli_stabilize(A, E, C.as_source_array(mu=mu), ast_spectrum, trans=False)
        KC = LowRankOperator(K, np.eye(len(K)), C.as_source_array(mu=mu))

        if not isinstance(D, ZeroOperator):
            BmKD = B - LowRankOperator(K, np.eye(len(K)), D.as_source_array(mu=mu))
        else:
            BmKD = B

        if self.dim_input <= self.dim_output:
            cf = solve_lyap_lrcf(A - KC, E, BmKD.as_range_array(mu=mu),
                                 trans=False, options=options_lrcf)
            return np.sqrt(self.C.apply(cf, mu=mu).norm2().sum())
        else:
            of = solve_lyap_lrcf(A - KC, E, C.as_source_array(mu=mu),
                                 trans=True, options=options_lrcf)
            return np.sqrt(BmKD.apply_adjoint(of, mu=mu).norm2().sum())

    @cached
    def linf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False):
        r"""Compute the L_infinity-norm of the |LTIModel|.

        The L-infinity norm of an |LTIModel| is defined via

        .. math::

            \lVert H \rVert_{\mathcal{L}_\infty}
            = \sup_{\omega \in \mathbb{R}}
            \lVert H(\boldsymbol{\imath} \omega) \rVert_2.

        Parameters
        ----------
        mu
            |Parameter|.
        return_fpeak
            Whether to return the frequency at which the maximum is achieved.
        ab13dd_equilibrate
            Whether `slycot.ab13dd` should use equilibration.

        Returns
        -------
        norm
            L_infinity-norm.
        fpeak
            Frequency at which the maximum is achieved (if `return_fpeak` is `True`).
        """
        if not config.HAVE_SLYCOT:
            raise NotImplementedError
        if not return_fpeak:
            return self.linf_norm(mu=mu, return_fpeak=True, ab13dd_equilibrate=ab13dd_equilibrate)[0]
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        A, B, C, D, E = (op.assemble(mu=mu) for op in [self.A, self.B, self.C, self.D, self.E])

        if self.order >= sparse_min_size():
            for op_name in ['A', 'B', 'C', 'D', 'E']:
                op = locals()[op_name]
                if not isinstance(op, NumpyMatrixOperator) or op.sparse:
                    self.logger.warning(f'Converting operator {op_name} to a NumPy array.')

        from slycot import ab13dd
        dico = 'C' if self.cont_time else 'D'
        jobe = 'I' if isinstance(self.E, IdentityOperator) else 'G'
        equil = 'S' if ab13dd_equilibrate else 'N'
        jobd = 'Z' if isinstance(self.D, ZeroOperator) else 'D'
        A, B, C, D, E = (to_matrix(op, format='dense') for op in [A, B, C, D, E])
        norm, fpeak = ab13dd(dico, jobe, equil, jobd,
                             self.order, self.dim_input, self.dim_output,
                             A, E, B, C, D)
        return norm, fpeak

    def get_ast_spectrum(self, ast_pole_data=None, mu=None):
        """Compute anti-stable subset of the poles of the |LTIModel|.

        Parameters
        ----------
        ast_pole_data
            Can be:

            - dictionary of parameters for :func:`~pymor.algorithms.eigs.eigs`,
            - list of anti-stable eigenvalues (scalars),
            - tuple `(lev, ew, rev)` where `ew` contains the sorted anti-stable eigenvalues
              and `lev` and `rev` are |VectorArrays| representing the eigenvectors.
            - `None` if anti-stable eigenvalues should be computed via dense methods.
        mu
            |Parameter|.

        Returns
        -------
        lev
            |VectorArray| of left eigenvectors.
        ew
            One-dimensional |NumPy array| of anti-stable eigenvalues sorted from smallest to
            largest.
        rev
            |VectorArray| of right eigenvectors.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        A, B, C, D, E = (op.assemble(mu=mu) for op in [self.A, self.B, self.C, self.D, self.E])

        if ast_pole_data is not None:
            if type(ast_pole_data) == dict:
                ew, rev = eigs(A, E=E if self.E else None, left_evp=False, **ast_pole_data)
                ast_idx = np.where(ew.real > 0.)
                ast_ews = ew[ast_idx]
                if len(ast_ews) == 0:
                    return self.solution_space.empty(), np.empty((0,)), self.solution_space.empty()

                ast_levs = A.source.empty(reserve=len(ast_ews))
                for ae in ast_ews:
                    # l=3 avoids issues with complex conjugate pairs
                    _, lev = eigs(A, E=E if self.E else None, k=1, l=3, sigma=ae, left_evp=True)
                    ast_levs.append(lev)
                return ast_levs, ast_ews, rev[ast_idx[0]]

            elif type(ast_pole_data) == list:
                assert all(np.real(ast_pole_data) > 0)
                ast_pole_data = np.sort(ast_pole_data)
                ast_levs = A.source.empty(reserve=len(ast_pole_data))
                ast_revs = A.source.empty(reserve=len(ast_pole_data))
                for ae in ast_pole_data:
                    _, lev = eigs(A, E=E if self.E else None, k=1, l=3, sigma=ae, left_evp=True)
                    ast_levs.append(lev)
                    _, rev = eigs(A, E=E if self.E else None, k=1, l=3, sigma=ae)
                    ast_revs.append(rev)
                return ast_levs, ast_pole_data, ast_revs

            elif type(ast_pole_data) == tuple:
                return ast_pole_data

            else:
                TypeError(f'ast_pole_data is of wrong type ({type(ast_pole_data)}).')

        else:
            if self.order >= sparse_min_size():
                if not isinstance(A, NumpyMatrixOperator) or A.sparse:
                    self.logger.warning('Converting operator A to a NumPy array.')
                if not isinstance(E, IdentityOperator):
                    if not isinstance(E, NumpyMatrixOperator) or E.sparse:
                        self.logger.warning('Converting operator E to a NumPy array.')

            A, E = (to_matrix(op, format='dense') for op in [A, E])
            ew, lev, rev = spla.eig(A, E if self.E else None, left=True)
            ast_idx = np.where(ew.real > 0.)
            ast_ews = ew[ast_idx]
            idx = ast_ews.argsort()

            ast_lev = self.A.source.from_numpy(lev[:, ast_idx][:, 0, :][:, idx].T)
            ast_rev = self.A.range.from_numpy(rev[:, ast_idx][:, 0, :][:, idx].T)

            return ast_lev, ast_ews[idx], ast_rev


class TransferFunction(InputOutputModel):
    """Class for systems represented by a transfer function.

    This class describes input-output systems given by a transfer
    function :math:`H(s, mu)`.

    Parameters
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    tf
        The transfer function defined at least on the open right complex half-plane.
        `tf(s, mu)` is a |NumPy array| of shape `(p, m)`.
    dtf
        The complex derivative of `H` with respect to `s`.
    cont_time
        `True` if the system is continuous-time, otherwise `False`.
    name
        Name of the system.

    Attributes
    ----------
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
    tf
        The transfer function.
    dtf
        The complex derivative of the transfer function.
    """

    def __init__(self, dim_input, dim_output, tf, dtf, parameters={}, cont_time=True, name=None):
        super().__init__(dim_input, dim_output, cont_time=cont_time, name=name)
        self.parameters_own = parameters
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of inputs:  {self.dim_input}\n'
            f'    number of outputs: {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    def eval_tf(self, s, mu=None):
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if not self.parametric:
            return self.tf(s)
        else:
            return self.tf(s, mu=mu)

    def eval_dtf(self, s, mu=None):
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if not self.parametric:
            return self.dtf(s)
        else:
            return self.dtf(s, mu=mu)

    def __add__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) + other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: self.eval_dtf(s, mu=mu) + other.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_input == other.dim_input
        assert self.dim_output == other.dim_output

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) - self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: other.eval_dtf(s, mu=mu) - self.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    def __neg__(self):
        tf = lambda s, mu=None: -self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: -self.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    def __mul__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_input == other.dim_input

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) @ other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: (self.eval_dtf(s, mu=mu) @ other.eval_tf(s, mu=mu)
                                  + self.eval_tf(s, mu=mu) @ other.eval_dtf(s, mu=mu))
        return self.with_(tf=tf, dtf=dtf)

    def __rmul__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.dim_output == other.dim_input

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) @ self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: (other.eval_dtf(s, mu=mu) @ self.eval_tf(s, mu=mu)
                                  + other.eval_tf(s, mu=mu) @ self.eval_dtf(s, mu=mu))
        return self.with_(tf=tf, dtf=dtf)


class SecondOrderModel(InputStateOutputModel):
    r"""Class for linear second order systems.

    This class describes input-output systems given by

    .. math::
        M(\mu) \ddot{x}(t, \mu)
        + E(\mu) \dot{x}(t, \mu)
        + K(\mu) x(t, \mu)
        & =
            B(\mu) u(t), \\
        y(t, \mu)
        & =
            C_p(\mu) x(t, \mu)
            + C_v(\mu) \dot{x}(t, \mu)
            + D(\mu) u(t),

    if continuous-time, or

    .. math::
        M(\mu) x(k + 2, \mu)
        + E(\mu) x(k + 1, \mu)
        + K(\mu) x(k, \mu)
        & =
            B(\mu) u(k), \\
        y(k, \mu)
        & =
            C_p(\mu) x(k, \mu)
            + C_v(\mu) x(k + 1, \mu)
            + D(\mu) u(k),

    if discrete-time, where :math:`M`, :math:`E`, :math:`K`, :math:`B`, :math:`C_p`, :math:`C_v`,
    and :math:`D` are linear operators.

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
    error_estimator
        An error estimator for the problem. This can be any object with an
        `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
        `estimate_error(U, mu)` method is added to the model which will call
        `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
        method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added to
        the model which forwards its arguments to the visualizer's `visualize` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system (equal to M.source.dim).
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
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
    """

    def __init__(self, M, E, K, B, Cp, Cv=None, D=None, cont_time=True,
                 solver_options=None, error_estimator=None, visualizer=None, name=None):

        assert M.linear and M.source == M.range
        assert E.linear and E.source == E.range == M.source
        assert K.linear and K.source == K.range == M.source
        assert B.linear and B.range == M.source
        assert Cp.linear and Cp.source == M.range

        Cv = Cv or ZeroOperator(Cp.range, Cp.source)
        assert Cv.linear and Cv.source == M.range and Cv.range == Cp.range

        D = D or ZeroOperator(Cp.range, B.source)
        assert D.linear and D.source == B.source and D.range == Cp.range

        assert solver_options is None or solver_options.keys() <= {'lyap_lrcf', 'lyap_dense'}

        super().__init__(B.source.dim, M.source, Cp.range.dim, cont_time=cont_time,
                         error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    second-order\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    @classmethod
    def from_matrices(cls, M, E, K, B, Cp, Cv=None, D=None, cont_time=True,
                      state_id='STATE', solver_options=None, error_estimator=None,
                      visualizer=None, name=None):
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
            The |NumPy array| or |SciPy spmatrix| Cv or `None` (then Cv is assumed to be zero).
        D
            The |NumPy array| or |SciPy spmatrix| D or `None` (then D is assumed to be zero).
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        error_estimator
            An error estimator for the problem. This can be any object with an
            `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
            `estimate_error(U, mu)` method is added to the model which will call
            `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        lti
            The SecondOrderModel with operators M, E, K, B, Cp, Cv, and D.
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
        B = NumpyMatrixOperator(B, range_id=state_id)
        Cp = NumpyMatrixOperator(Cp, source_id=state_id)
        if Cv is not None:
            Cv = NumpyMatrixOperator(Cv, source_id=state_id)
        if D is not None:
            D = NumpyMatrixOperator(D)

        return cls(M, E, K, B, Cp, Cv, D, cont_time=cont_time,
                   solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer, name=name)

    def to_matrices(self):
        """Return operators as matrices.

        Returns
        -------
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
            The |NumPy array| or |SciPy spmatrix| Cv or `None` (if Cv is a `ZeroOperator`).
        D
            The |NumPy array| or |SciPy spmatrix| D or `None` (if D is a `ZeroOperator`).
        """
        M = to_matrix(self.M)
        E = to_matrix(self.E)
        K = to_matrix(self.K)
        B = to_matrix(self.B)
        Cp = to_matrix(self.Cp)
        Cv = None if isinstance(self.Cv, ZeroOperator) else to_matrix(self.Cv)
        D = None if isinstance(self.D, ZeroOperator) else to_matrix(self.D)
        return M, E, K, B, Cp, Cv, D

    @classmethod
    def from_files(cls, M_file, E_file, K_file, B_file, Cp_file, Cv_file=None, D_file=None, cont_time=True,
                   state_id='STATE', solver_options=None, error_estimator=None, visualizer=None,
                   name=None):
        """Create |LTIModel| from matrices stored in separate files.

        Parameters
        ----------
        M_file
            The name of the file (with extension) containing A.
        E_file
            The name of the file (with extension) containing E.
        K_file
            The name of the file (with extension) containing K.
        B_file
            The name of the file (with extension) containing B.
        Cp_file
            The name of the file (with extension) containing Cp.
        Cv_file
            `None` or the name of the file (with extension) containing Cv.
        D_file
            `None` or the name of the file (with extension) containing D.
        cont_time
            `True` if the system is continuous-time, otherwise `False`.
        state_id
            Id of the state space.
        solver_options
            The solver options to use to solve the Lyapunov equations.
        error_estimator
            An error estimator for the problem. This can be any object with an
            `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
            `estimate_error(U, mu)` method is added to the model which will call
            `error_estimator.estimate_error(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        some
            The |SecondOrderModel| with operators M, E, K, B, Cp, Cv, and D.
        """
        from pymor.tools.io import load_matrix

        M = load_matrix(M_file)
        E = load_matrix(E_file)
        K = load_matrix(K_file)
        B = load_matrix(B_file)
        Cp = load_matrix(Cp_file)
        Cv = load_matrix(Cv_file) if Cv_file is not None else None
        D = load_matrix(D_file) if D_file is not None else None

        return cls.from_matrices(M, E, K, B, Cp, Cv, D, cont_time=cont_time,
                                 state_id=state_id, solver_options=solver_options,
                                 error_estimator=error_estimator, visualizer=visualizer, name=name)

    def to_files(self, M_file, E_file, K_file, B_file, Cp_file, Cv_file=None, D_file=None):
        """Write operators to files as matrices.

        Parameters
        ----------
        M_file
            The name of the file (with extension) containing M.
        E_file
            The name of the file (with extension) containing E.
        K_file
            The name of the file (with extension) containing K.
        B_file
            The name of the file (with extension) containing B.
        Cp_file
            The name of the file (with extension) containing Cp.
        Cv_file
            The name of the file (with extension) containing Cv or `None` if D is a `ZeroOperator`.
        D_file
            The name of the file (with extension) containing D or `None` if D is a `ZeroOperator`.
        """
        if Cv_file is None and not isinstance(self.Cv, ZeroOperator):
            raise ValueError('Cv is not zero, Cv_file must be given')
        if D_file is None and not isinstance(self.D, ZeroOperator):
            raise ValueError('D is not zero, D_file must be given')

        from pymor.tools.io import save_matrix

        M, E, K, B, Cp, Cv, D = self.to_matrices()
        for mat, file in [(M, M_file), (E, E_file), (K, K_file),
                          (B, B_file), (Cp, Cp_file), (Cv, Cv_file), (D, D_file)]:
            if mat is None:
                continue
            save_matrix(file, mat)

    @cached
    def to_lti(self):
        r"""Return a first order representation.

        The first order representation

        .. math::
            \begin{bmatrix}
                I & 0 \\
                0 & M
            \end{bmatrix}
            \frac{\mathrm{d}}{\mathrm{d}t}\!
            \begin{bmatrix}
                x(t) \\
                \dot{x}(t)
            \end{bmatrix}
            & =
            \begin{bmatrix}
                0 & I \\
                -K & -E
            \end{bmatrix}
            \begin{bmatrix}
                x(t) \\
                \dot{x}(t)
            \end{bmatrix}
            +
            \begin{bmatrix}
                0 \\
                B
            \end{bmatrix}
            u(t), \\
            y(t)
            & =
            \begin{bmatrix}
                C_p & C_v
            \end{bmatrix}
            \begin{bmatrix}
                x(t) \\
                \dot{x}(t)
            \end{bmatrix}
            + D u(t)

        is returned.

        Returns
        -------
        lti
            |LTIModel| equivalent to the second-order model.
        """
        return LTIModel(A=SecondOrderModelOperator(0, 1, -self.E, -self.K),
                        B=BlockColumnOperator([ZeroOperator(self.B.range, self.B.source), self.B]),
                        C=BlockRowOperator([self.Cp, self.Cv]),
                        D=self.D,
                        E=(IdentityOperator(BlockVectorSpace([self.M.source, self.M.source]))
                           if isinstance(self.M, IdentityOperator) else
                           BlockDiagonalOperator([IdentityOperator(self.M.source), self.M])),
                        cont_time=self.cont_time,
                        solver_options=self.solver_options,
                        error_estimator=self.error_estimator,
                        visualizer=self.visualizer,
                        name=self.name + '_first_order')

    def __add__(self, other):
        """Add a |SecondOrderModel| or an |LTIModel|."""
        if isinstance(other, LTIModel):
            return self.to_lti() + other

        if not isinstance(other, SecondOrderModel):
            return NotImplemented

        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.source
        assert self.D.range == other.D.range

        M = BlockDiagonalOperator([self.M, other.M])
        E = BlockDiagonalOperator([self.E, other.E])
        K = BlockDiagonalOperator([self.K, other.K])
        B = BlockColumnOperator([self.B, other.B])
        Cp = BlockRowOperator([self.Cp, other.Cp])
        Cv = BlockRowOperator([self.Cv, other.Cv])
        D = self.D + other.D
        return self.with_(M=M, E=E, K=K, B=B, Cp=Cp, Cv=Cv, D=D)

    def __radd__(self, other):
        """Add to an |LTIModel|."""
        if isinstance(other, LTIModel):
            return other + self.to_lti()
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtract a |SecondOrderModel| or an |LTIModel|."""
        return self + (-other)

    def __rsub__(self, other):
        """Subtract from an |LTIModel|."""
        if isinstance(other, LTIModel):
            return other - self.to_lti()
        else:
            return NotImplemented

    def __neg__(self):
        """Negate the |SecondOrderModel|."""
        return self.with_(Cp=-self.Cp, Cv=-self.Cv, D=-self.D)

    def __mul__(self, other):
        """Postmultiply by a |SecondOrderModel| or an |LTIModel|."""
        if isinstance(other, LTIModel):
            return self.to_lti() * other

        if not isinstance(other, SecondOrderModel):
            return NotImplemented

        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.range

        M = BlockDiagonalOperator([self.M, other.M])
        E = BlockOperator([[self.E, -(self.B @ other.Cv)],
                           [None, other.E]])
        K = BlockOperator([[self.K, -(self.B @ other.Cp)],
                           [None, other.K]])
        B = BlockColumnOperator([self.B @ other.D, other.B])
        Cp = BlockRowOperator([self.Cp, self.D @ other.Cp])
        Cv = BlockRowOperator([self.Cv, self.D @ other.Cv])
        D = self.D @ other.D
        return self.with_(M=M, E=E, K=K, B=B, Cp=Cp, Cv=Cv, D=D)

    def __rmul__(self, other):
        """Premultiply by an |LTIModel|."""
        if isinstance(other, LTIModel):
            return other * self.to_lti()
        else:
            return NotImplemented

    @cached
    def poles(self, mu=None):
        """Compute system poles.

        .. note::
            Assumes the systems is small enough to use a dense eigenvalue solver.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        One-dimensional |NumPy array| of system poles.
        """
        return self.to_lti().poles(mu=mu)

    def eval_tf(self, s, mu=None):
        r"""Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            (C_p(\mu) + s C_v(\mu))
                (s^2 M(\mu) + s E(\mu) + K(\mu))^{-1} B(\mu)
            + D(\mu).

        .. note::
            Assumes that the number of inputs and outputs is much smaller than the order of the
            system.

        Parameters
        ----------
        s
            Complex number.
        mu
            |Parameter values|.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array| of shape
            `(self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        M = self.M
        E = self.E
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv
        D = self.D

        s2MpsEpK = s**2 * M + s * E + K
        if self.dim_input <= self.dim_output:
            CppsCv = Cp + s * Cv
            tfs = CppsCv.apply(s2MpsEpK.apply_inverse(B.as_range_array(mu=mu),
                                                      mu=mu),
                               mu=mu).to_numpy().T
        else:
            tfs = B.apply_adjoint(
                s2MpsEpK.apply_inverse_adjoint(
                    Cp.as_source_array(mu=mu) + Cv.as_source_array(mu=mu) * s.conjugate(),
                    mu=mu),
                mu=mu).to_numpy().conj()
        if not isinstance(D, ZeroOperator):
            tfs += to_matrix(D, format='dense')
        return tfs

    def eval_dtf(self, s, mu=None):
        r"""Evaluate the derivative of the transfer function.

        .. math::
            s C_v(\mu) (s^2 M(\mu) + s E(\mu) + K(\mu))^{-1} B(\mu)
            - (C_p(\mu) + s C_v(\mu))
                (s^2 M(\mu) + s E(\mu) + K(\mu))^{-1}
                (2 s M(\mu) + E(\mu))
                (s^2 M(\mu) + s E(\mu) + K(\mu))^{-1}
                B(\mu).

        .. note::
            Assumes that the number of inputs and outputs is much smaller than the order of the
            system.

        Parameters
        ----------
        s
            Complex number.
        mu
            |Parameter values|.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`, |NumPy array| of
            shape `(self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        M = self.M
        E = self.E
        K = self.K
        B = self.B
        Cp = self.Cp
        Cv = self.Cv

        s2MpsEpK = (s**2 * M + s * E + K).assemble(mu=mu)
        sM2pE = 2 * s * M + E
        if self.dim_input <= self.dim_output:
            dtfs = Cv.apply(s2MpsEpK.apply_inverse(B.as_range_array(mu=mu)),
                            mu=mu).to_numpy().T * s
            CppsCv = Cp + s * Cv
            dtfs -= CppsCv.apply(
                s2MpsEpK.apply_inverse(
                    sM2pE.apply(
                        s2MpsEpK.apply_inverse(
                            B.as_range_array(mu=mu)),
                        mu=mu)),
                mu=mu).to_numpy().T
        else:
            dtfs = B.apply_adjoint(s2MpsEpK.apply_inverse_adjoint(Cv.as_source_array(mu=mu)),
                                   mu=mu).to_numpy().conj() * s
            dtfs -= B.apply_adjoint(
                s2MpsEpK.apply_inverse_adjoint(
                    sM2pE.apply_adjoint(
                        s2MpsEpK.apply_inverse_adjoint(
                            Cp.as_source_array(mu=mu) + Cv.as_source_array(mu=mu) * s.conjugate()),
                        mu=mu)),
                mu=mu).to_numpy().conj()
        return dtfs

    @cached
    def gramian(self, typ, mu=None):
        """Compute a second-order Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'pc_lrcf'`: low-rank Cholesky factor of the position controllability Gramian,
            - `'vc_lrcf'`: low-rank Cholesky factor of the velocity controllability Gramian,
            - `'po_lrcf'`: low-rank Cholesky factor of the position observability Gramian,
            - `'vo_lrcf'`: low-rank Cholesky factor of the velocity observability Gramian,
            - `'pc_dense'`: dense position controllability Gramian,
            - `'vc_dense'`: dense velocity controllability Gramian,
            - `'po_dense'`: dense position observability Gramian,
            - `'vo_dense'`: dense velocity observability Gramian.

            .. note::
                For `'*_lrcf'` types, the method assumes the system is asymptotically stable.
                For `'*_dense'` types, the method assumes there are no two system poles which add to
                zero.
        mu
            |Parameter values|.

        Returns
        -------
        If typ is `'pc_lrcf'`, `'vc_lrcf'`, `'po_lrcf'` or `'vo_lrcf'`, then the Gramian factor as a
        |VectorArray| from `self.M.source`.
        If typ is `'pc_dense'`, `'vc_dense'`, `'po_dense'` or `'vo_dense'`, then the Gramian as a
        |NumPy array|.
        """
        assert typ in ('pc_lrcf', 'vc_lrcf', 'po_lrcf', 'vo_lrcf',
                       'pc_dense', 'vc_dense', 'po_dense', 'vo_dense')

        if typ.endswith('lrcf'):
            return self.to_lti().gramian(typ[1:], mu=mu).block(0 if typ.startswith('p') else 1)
        else:
            g = self.to_lti().gramian(typ[1:], mu=mu)
            if typ.startswith('p'):
                return g[:self.order, :self.order]
            else:
                return g[self.order:, self.order:]

    def psv(self, mu=None):
        """Position singular values.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(
            self.gramian('po_lrcf', mu=mu)[:self.order]
            .inner(self.gramian('pc_lrcf', mu=mu)[:self.order])
        )

    def vsv(self, mu=None):
        """Velocity singular values.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(
            self.gramian('vo_lrcf', mu=mu)[:self.order]
            .inner(self.gramian('vc_lrcf', mu=mu)[:self.order], product=self.M)
        )

    def pvsv(self, mu=None):
        """Position-velocity singular values.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(
            self.gramian('vo_lrcf', mu=mu)[:self.order]
            .inner(self.gramian('pc_lrcf', mu=mu)[:self.order], product=self.M)
        )

    def vpsv(self, mu=None):
        """Velocity-position singular values.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        One-dimensional |NumPy array| of singular values.
        """
        return spla.svdvals(
            self.gramian('po_lrcf', mu=mu)[:self.order]
            .inner(self.gramian('vc_lrcf', mu=mu)[:self.order])
        )

    @cached
    def h2_norm(self, mu=None):
        """Compute the H2-norm.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        norm
            H_2-norm.
        """
        return self.to_lti().h2_norm(mu=mu)

    @cached
    def hinf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False):
        """Compute the H_infinity-norm.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.
        return_fpeak
            Should the frequency at which the maximum is achieved should be returned.
        ab13dd_equilibrate
            Should `slycot.ab13dd` use equilibration.

        Returns
        -------
        norm
            H_infinity-norm.
        fpeak
            Frequency at which the maximum is achieved (if `return_fpeak` is `True`).
        """
        return self.to_lti().hinf_norm(mu=mu,
                                       return_fpeak=return_fpeak,
                                       ab13dd_equilibrate=ab13dd_equilibrate)

    @cached
    def hankel_norm(self, mu=None):
        """Compute the Hankel-norm.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        norm
            Hankel-norm.
        """
        return self.to_lti().hankel_norm(mu=mu)


class LinearDelayModel(InputStateOutputModel):
    r"""Class for linear delay systems.

    This class describes input-state-output systems given by

    .. math::
        E x'(t)
        & =
            A x(t)
            + \sum_{i = 1}^q{A_i x(t - \tau_i)}
            + B u(t), \\
        y(t)
        & =
            C x(t)
            + D u(t),

    if continuous-time, or

    .. math::
        E x(k + 1)
        & =
            A x(k)
            + \sum_{i = 1}^q{A_i x(k - \tau_i)}
            + B u(k), \\
        y(k)
        & =
            C x(k)
            + D u(k),

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`, :math:`B`, :math:`C`, and :math:`D`
    are linear operators.

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
    error_estimator
        An error estimator for the problem. This can be any object with an
        `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
        `estimate_error(U, mu)` method is added to the model which will call
        `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
        method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added to
        the model which forwards its arguments to the visualizer's `visualize` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system (equal to A.source.dim).
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
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
    """

    def __init__(self, A, Ad, tau, B, C, D=None, E=None, cont_time=True,
                 error_estimator=None, visualizer=None, name=None):

        assert A.linear and A.source == A.range
        assert isinstance(Ad, tuple) and len(Ad) > 0
        assert all(Ai.linear and Ai.source == Ai.range == A.source for Ai in Ad)
        assert isinstance(tau, tuple) and len(tau) == len(Ad) and all(taui > 0 for taui in tau)
        assert B.linear and B.range == A.source
        assert C.linear and C.source == A.range

        D = D or ZeroOperator(C.range, B.source)
        assert D.linear and D.source == B.source and D.range == C.range

        E = E or IdentityOperator(A.source)
        assert E.linear and E.source == E.range == A.source

        super().__init__(B.source.dim, A.source, C.range.dim, cont_time=cont_time,
                         error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.q = len(Ad)
        self.solution_space = A.source

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    time-delay\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    def __add__(self, other):
        """Add an |LTIModel|, |SecondOrderModel| or |LinearDelayModel|."""
        if isinstance(other, SecondOrderModel):
            other = other.to_lti()

        if isinstance(other, LTIModel):
            Ad = tuple(BlockDiagonalOperator([op, ZeroOperator(other.solution_space, other.solution_space)])
                       for op in self.Ad)
            tau = self.tau
        elif isinstance(other, LinearDelayModel):
            tau = tuple(set(self.tau).union(set(other.tau)))
            Ad = [None for _ in tau]
            for i, taui in enumerate(tau):
                if taui in self.tau and taui in other.tau:
                    Ad[i] = BlockDiagonalOperator([self.Ad[self.tau.index(taui)],
                                                   other.Ad[other.tau.index(taui)]])
                elif taui in self.tau:
                    Ad[i] = BlockDiagonalOperator([self.Ad[self.tau.index(taui)],
                                                   ZeroOperator(other.solution_space, other.solution_space)])
                else:
                    Ad[i] = BlockDiagonalOperator([ZeroOperator(self.solution_space, self.solution_space),
                                                   other.Ad[other.tau.index(taui)]])
            Ad = tuple(Ad)
        else:
            return NotImplemented

        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.source
        assert self.D.range == other.D.range

        E = BlockDiagonalOperator([self.E, other.E])
        A = BlockDiagonalOperator([self.A, other.A])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, other.C])
        D = self.D + other.D
        return self.with_(E=E, A=A, Ad=Ad, tau=tau, B=B, C=C, D=D)

    def __radd__(self, other):
        """Add to an |LTIModel| or a |SecondOrderModel|."""
        if isinstance(other, LTIModel):
            return self + other
        elif isinstance(other, SecondOrderModel):
            return self + other.to_lti()
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtract an |LTIModel|, |SecondOrderModel| or |LinearDelayModel|."""
        return self + (-other)

    def __rsub__(self, other):
        """Subtract from an |LTIModel| or a |SecondOrderModel|."""
        if isinstance(other, (LTIModel, SecondOrderModel)):
            return -(self - other)
        else:
            return NotImplemented

    def __neg__(self):
        """Negate the |LinearDelayModel|."""
        return self.with_(C=-self.C, D=-self.D)

    def __mul__(self, other):
        """Postmultiply an |LTIModel|, |SecondOrderModel| or |LinearDelayModel|."""
        if isinstance(other, SecondOrderModel):
            other = other.to_lti()

        if isinstance(other, LTIModel):
            Ad = tuple(BlockDiagonalOperator([op, ZeroOperator(other.solution_space, other.solution_space)])
                       for op in self.Ad)
            tau = self.tau
        elif isinstance(other, LinearDelayModel):
            tau = tuple(set(self.tau).union(set(other.tau)))
            Ad = [None for _ in tau]
            for i, taui in enumerate(tau):
                if taui in self.tau and taui in other.tau:
                    Ad[i] = BlockDiagonalOperator([self.Ad[self.tau.index(taui)],
                                                   other.Ad[other.tau.index(taui)]])
                elif taui in self.tau:
                    Ad[i] = BlockDiagonalOperator([self.Ad[self.tau.index(taui)],
                                                   ZeroOperator(other.solution_space, other.solution_space)])
                else:
                    Ad[i] = BlockDiagonalOperator([ZeroOperator(self.solution_space, self.solution_space),
                                                   other.Ad[other.tau.index(taui)]])
            Ad = tuple(Ad)
        else:
            return NotImplemented

        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.range

        E = BlockDiagonalOperator([self.E, other.E])
        A = BlockOperator([[self.A, self.B @ other.C],
                           [None, other.A]])
        B = BlockColumnOperator([self.B @ other.D, other.B])
        C = BlockRowOperator([self.C, self.D @ other.C])
        D = self.D @ other.D
        return self.with_(E=E, A=A, Ad=Ad, tau=tau, B=B, C=C, D=D)

    def __rmul__(self, other):
        """Premultiply by an |LTIModel| or a |SecondOrderModel|."""
        assert self.cont_time == other.cont_time
        assert self.D.source == other.D.range

        if isinstance(other, SecondOrderModel):
            other = other.to_lti()

        if isinstance(other, LTIModel):
            E = BlockDiagonalOperator([other.E, self.E])
            A = BlockOperator([[other.A, other.B @ self.C],
                               [None, self.A]])
            Ad = tuple(BlockDiagonalOperator([ZeroOperator(other.solution_space, other.solution_space), op])
                       for op in self.Ad)
            B = BlockColumnOperator([other.B @ self.D, self.B])
            C = BlockRowOperator([other.C, other.D @ self.C])
            D = other.D @ self.D
            return self.with_(E=E, A=A, Ad=Ad, B=B, C=C, D=D)
        else:
            return NotImplemented

    def eval_tf(self, s, mu=None):
        r"""Evaluate the transfer function.

        The transfer function at :math:`s` is

        .. math::
            C \left(s E - A
                - \sum_{i = 1}^q{e^{-\tau_i s} A_i}\right)^{-1} B
            + D.

        .. note::
            Assumes that the number of inputs and outputs is much smaller than the order of the
            system.

        Parameters
        ----------
        s
            Complex number.
        mu
            |Parameter values|.

        Returns
        -------
        tfs
            Transfer function evaluated at the complex number `s`, |NumPy array| of shape
            `(self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C
        D = self.D
        E = self.E

        middle = LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        if self.dim_input <= self.dim_output:
            tfs = C.apply(middle.apply_inverse(B.as_range_array(mu=mu),
                                               mu=mu),
                          mu=mu).to_numpy().T
        else:
            tfs = B.apply_adjoint(middle.apply_inverse_adjoint(C.as_source_array(mu=mu),
                                                               mu=mu),
                                  mu=mu).to_numpy().conj()
        if not isinstance(D, ZeroOperator):
            tfs += to_matrix(D, format='dense')
        return tfs

    def eval_dtf(self, s, mu=None):
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
            Assumes that the number of inputs and outputs is much smaller than the order of the
            system.

        Parameters
        ----------
        s
            Complex number.
        mu
            |Parameter values|.

        Returns
        -------
        dtfs
            Derivative of transfer function evaluated at the complex number `s`, |NumPy array| of
            shape `(self.dim_output, self.dim_input)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A
        Ad = self.Ad
        B = self.B
        C = self.C
        E = self.E

        left_and_right = LincombOperator((E, A) + Ad,
                                         (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau)).assemble(mu=mu)
        middle = LincombOperator((E,) + Ad, (1,) + tuple(taui * np.exp(-taui * s) for taui in self.tau))
        if self.dim_input <= self.dim_output:
            dtfs = -C.apply(
                left_and_right.apply_inverse(
                    middle.apply(left_and_right.apply_inverse(B.as_range_array(mu=mu)),
                                 mu=mu)),
                mu=mu).to_numpy().T
        else:
            dtfs = -B.apply_adjoint(
                left_and_right.apply_inverse_adjoint(
                    middle.apply_adjoint(left_and_right.apply_inverse_adjoint(C.as_source_array(mu=mu)),
                                         mu=mu)),
                mu=mu).to_numpy().conj()
        return dtfs


class LinearStochasticModel(InputStateOutputModel):
    r"""Class for linear stochastic systems.

    This class describes input-state-output systems given by

    .. math::
        E \mathrm{d}x(t)
        & =
            A x(t) \mathrm{d}t
            + \sum_{i = 1}^q{A_i x(t) \mathrm{d}\omega_i(t)}
            + B u(t) \mathrm{d}t, \\
        y(t)
        & =
            C x(t)
            + D u(t),

    if continuous-time, or

    .. math::
        E x(k + 1)
        & =
            A x(k)
            + \sum_{i = 1}^q{A_i x(k) \omega_i(k)}
            + B u(k), \\
        y(k)
        & =
            C x(k)
            + D u(t),

    if discrete-time, where :math:`E`, :math:`A`, :math:`A_i`, :math:`B`, :math:`C`, and :math:`D`
    are linear operators and :math:`\omega_i` are stochastic processes.

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
    error_estimator
        An error estimator for the problem. This can be any object with an
        `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
        `estimate_error(U, mu)` method is added to the model which will call
        `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
        method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added to
        the model which forwards its arguments to the visualizer's `visualize` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system (equal to A.source.dim).
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
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
    """

    def __init__(self, A, As, B, C, D=None, E=None, cont_time=True,
                 error_estimator=None, visualizer=None, name=None):

        assert A.linear and A.source == A.range
        assert isinstance(As, tuple) and len(As) > 0
        assert all(Ai.linear and Ai.source == Ai.range == A.source for Ai in As)
        assert B.linear and B.range == A.source
        assert C.linear and C.source == A.range

        D = D or ZeroOperator(C.range, B.source)
        assert D.linear and D.source == B.source and D.range == C.range

        E = E or IdentityOperator(A.source)
        assert E.linear and E.source == E.range == A.source

        assert cont_time in (True, False)

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.q = len(As)

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    stochastic\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )


class BilinearModel(InputStateOutputModel):
    r"""Class for bilinear systems.

    This class describes input-output systems given by

    .. math::
        E x'(t)
        & =
            A x(t)
            + \sum_{i = 1}^m{N_i x(t) u_i(t)}
            + B u(t), \\
        y(t)
        & =
            C x(t)
            + D u(t),

    if continuous-time, or

    .. math::
        E x(k + 1)
        & =
            A x(k)
            + \sum_{i = 1}^m{N_i x(k) u_i(k)}
            + B u(k), \\
        y(k)
        & =
            C x(k)
            + D u(t),

    if discrete-time, where :math:`E`, :math:`A`, :math:`N_i`, :math:`B`, :math:`C`, and :math:`D`
    are linear operators and :math:`m` is the number of inputs.

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
    error_estimator
        An error estimator for the problem. This can be any object with an
        `estimate_error(U, mu, model)` method. If `error_estimator` is not `None`, an
        `estimate_error(U, mu)` method is added to the model which will call
        `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
        method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added to
        the model which forwards its arguments to the visualizer's `visualize` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system (equal to A.source.dim).
    dim_input
        The number of inputs.
    dim_output
        The number of outputs.
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
    """

    def __init__(self, A, N, B, C, D, E=None, cont_time=True,
                 error_estimator=None, visualizer=None, name=None):

        assert A.linear and A.source == A.range
        assert B.linear and B.range == A.source
        assert isinstance(N, tuple) and len(N) == B.source.dim
        assert all(Ni.linear and Ni.source == Ni.range == A.source for Ni in N)
        assert C.linear and C.source == A.range

        D = D or ZeroOperator(C.range, B.source)
        assert D.linear and D.source == B.source and D.range == C.range

        E = E or IdentityOperator(A.source)
        assert E.linear and E.source == E.range == A.source

        assert cont_time in (True, False)

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         error_estimator=error_estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.linear = False

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    bilinear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )


def _lti_to_poles_b_c(lti):
    """Compute poles and residues.

    Parameters
    ----------
    lti
        |LTIModel| consisting of |Operators| that can be converted to |NumPy arrays|.
        The D operator is ignored.

    Returns
    -------
    poles
        1D |NumPy array| of poles.
    b
        |NumPy array| of shape `(lti.order, lti.dim_input)`.
    c
        |NumPy array| of shape `(lti.order, lti.dim_output)`.
    """
    A = to_matrix(lti.A, format='dense')
    B = to_matrix(lti.B, format='dense')
    C = to_matrix(lti.C, format='dense')
    if isinstance(lti.E, IdentityOperator):
        poles, X = spla.eig(A)
        EX = X
    else:
        E = to_matrix(lti.E, format='dense')
        poles, X = spla.eig(A, E)
        EX = E @ X
    b = spla.solve(EX, B)
    c = (C @ X).T
    return poles, b, c


def _poles_b_c_to_lti(poles, b, c):
    r"""Create an |LTIModel| from poles and residue rank-1 factors.

    Returns an |LTIModel| with real matrices such that its transfer
    function is

    .. math::
        \sum_{i = 1}^r \frac{c_i b_i^T}{s - \lambda_i}

    where :math:`\lambda_i, b_i, c_i` are the poles and residue rank-1
    factors.

    Parameters
    ----------
    poles
        Sequence of poles.
    b
        |NumPy array| of shape `(rom.order, rom.dim_input)`.
    c
        |NumPy array| of shape `(rom.order, rom.dim_output)`.

    Returns
    -------
    |LTIModel|.
    """
    A, B, C = [], [], []
    for i, pole in enumerate(poles):
        if pole.imag == 0:
            A.append(pole.real)
            B.append(b[i].real)
            C.append(c[i].real[:, np.newaxis])
        elif pole.imag > 0:
            A.append([[pole.real, pole.imag],
                      [-pole.imag, pole.real]])
            B.append(np.vstack([2 * b[i].real, -2 * b[i].imag]))
            C.append(np.hstack([c[i].real[:, np.newaxis], c[i].imag[:, np.newaxis]]))
    A = spla.block_diag(*A)
    B = np.vstack(B)
    C = np.hstack(C)
    return LTIModel.from_matrices(A, B, C)
