# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.lyapunov import solve_lyap_lrcf, solve_lyap_dense
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import cached
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.models.interface import Model
from pymor.operators.block import (BlockOperator, BlockRowOperator, BlockColumnOperator, BlockDiagonalOperator,
                                   SecondOrderModelOperator)
from pymor.operators.constructions import IdentityOperator, LincombOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu, Parameters
from pymor.tools.formatrepr import indent_value
from pymor.vectorarrays.block import BlockVectorSpace


@defaults('value')
def sparse_min_size(value=1000):
    """Return minimal sparse problem size for which to warn about converting to dense."""
    return value


class InputOutputModel(Model):
    """Base class for input-output systems."""

    cache_region = 'memory'

    def __init__(self, input_space, output_space, cont_time=True,
                 estimator=None, visualizer=None, name=None):
        assert cont_time in (True, False)
        super().__init__(estimator=estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    @property
    def input_dim(self):
        return self.input_space.dim

    @property
    def output_dim(self):
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
            `(len(w), self.output_dim, self.input_dim)`.
        """
        if not self.cont_time:
            raise NotImplementedError
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        return np.stack([self.eval_tf(1j * wi, mu=mu) for wi in w])

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
            out = ax.semilogx(freq, 20 * np.log2(mag), **mpl_kwargs)
        else:
            out = ax.loglog(freq, mag, **mpl_kwargs)

        ax.set_title('Magnitude plot')
        freq_unit = ' (Hz)' if Hz else ' (rad/s)'
        ax.set_xlabel('Frequency' + freq_unit)
        mag_unit = ' (dB)' if dB else ''
        ax.set_ylabel('Magnitude' + mag_unit)

        return out


class InputStateOutputModel(InputOutputModel):
    """Base class for input-output systems with state space."""

    def __init__(self, input_space, solution_space, output_space, cont_time=True,
                 estimator=None, visualizer=None, name=None):
        super().__init__(input_space, output_space, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer, name=name)
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
    estimator
        An error estimator for the problem. This can be any object with an `estimate(U, mu, model)`
        method. If `estimator` is not `None`, an `estimate(U, mu)` method is added to the model
        which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
        method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added to
        the model which forwards its arguments to the visualizer's `visualize` method.
    name
        Name of the system.

    Attributes
    ----------
    order
        The order of the system.
    input_dim
        The number of inputs.
    output_dim
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
                 solver_options=None, estimator=None, visualizer=None, name=None):

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

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.input_dim}\n'
            f'    number of outputs:   {self.output_dim}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    @classmethod
    def from_matrices(cls, A, B, C, D=None, E=None, cont_time=True,
                      state_id='STATE', solver_options=None, estimator=None,
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
        estimator
            An error estimator for the problem. This can be any object with an
            `estimate(U, mu, model)` method. If `estimator` is not `None`, an `estimate(U, mu)`
            method is added to the model which will call `estimator.estimate(U, mu, self)`.
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
                   solver_options=solver_options, estimator=estimator, visualizer=visualizer,
                   name=name)

    @classmethod
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, cont_time=True,
                   state_id='STATE', solver_options=None, estimator=None, visualizer=None,
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
        estimator
            An error estimator for the problem. This can be any object with an
            `estimate(U, mu, model)` method. If `estimator` is not `None`, an `estimate(U, mu)`
            method is added to the model which will call `estimator.estimate(U, mu, self)`.
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
                                 estimator=estimator, visualizer=visualizer, name=name)

    @classmethod
    def from_mat_file(cls, file_name, cont_time=True,
                      state_id='STATE', solver_options=None, estimator=None,
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
        estimator
            An error estimator for the problem. This can be any object with an
            `estimate(U, mu, model)` method. If `estimator` is not `None`, an `estimate(U, mu)`
            method is added to the model which will call `estimator.estimate(U, mu, self)`.
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
                                 estimator=estimator, visualizer=visualizer, name=name)

    @classmethod
    def from_abcde_files(cls, files_basename, cont_time=True,
                         state_id='STATE', solver_options=None, estimator=None,
                         visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in a .[ABCDE] files.

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
        estimator
            An error estimator for the problem. This can be any object with an
            `estimate(U, mu, model)` method. If `estimator` is not `None`, an `estimate(U, mu)`
            method is added to the model which will call `estimator.estimate(U, mu, self)`.
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
                                 estimator=estimator, visualizer=visualizer, name=name)

    def __add__(self, other):
        """Add an |LTIModel|."""
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

        if not isinstance(other, LTIModel):
            return NotImplemented

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
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

        if not isinstance(other, LTIModel):
            return NotImplemented

        A = BlockDiagonalOperator([self.A, other.A])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, -other.C])
        if self.D is other.D:
            D = ZeroOperator(self.output_space, self.input_space)
        else:
            D = self.D - other.D
        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            E = BlockDiagonalOperator([self.E, other.E])
        return self.with_(A=A, B=B, C=C, D=D, E=E)

    def __neg__(self):
        """Negate the |LTIModel|."""
        return self.with_(C=-self.C, D=-self.D)

    def __mul__(self, other):
        """Postmultiply by an |LTIModel|."""
        assert self.cont_time == other.cont_time
        assert self.input_space == other.output_space

        if not isinstance(other, LTIModel):
            return NotImplemented

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
            Assumes that either the number of inputs or the number of outputs is much smaller than
            the order of the system.

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
            `(self.output_dim, self.input_dim)`.
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
        if self.input_dim <= self.output_dim:
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
            Assumes that either the number of inputs or the number of outputs is much smaller than
            the order of the system.

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
            shape `(self.output_dim, self.input_dim)`.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        A = self.A
        B = self.B
        C = self.C
        E = self.E

        sEmA = (s * E - A).assemble(mu=mu)
        if self.input_dim <= self.output_dim:
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
        assert self.parameters.assert_compatible(mu)
        if self.input_dim <= self.output_dim:
            cf = self.gramian('c_lrcf', mu=mu)
            return np.sqrt(self.C.apply(cf, mu=mu).l2_norm2().sum())
        else:
            of = self.gramian('o_lrcf', mu=mu)
            return np.sqrt(self.B.apply_adjoint(of, mu=mu).l2_norm2().sum())

    @cached
    def hinf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False):
        """Compute the H_infinity-norm of the |LTIModel|.

        .. note::
            Assumes the system is asymptotically stable.

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
        if not config.HAVE_SLYCOT:
            raise NotImplementedError
        if not return_fpeak:
            return self.hinf_norm(mu=mu, return_fpeak=True, ab13dd_equilibrate=ab13dd_equilibrate)[0]
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
                             self.order, self.input_dim, self.output_dim,
                             A, E, B, C, D)
        return norm, fpeak

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


class TransferFunction(InputOutputModel):
    """Class for systems represented by a transfer function.

    This class describes input-output systems given by a transfer
    function :math:`H(s, mu)`.

    Parameters
    ----------
    input_space
        The input |VectorSpace|. Typically `NumpyVectorSpace(m)` where m is the number of inputs.
    output_space
        The output |VectorSpace|. Typically `NumpyVectorSpace(p)` where p is the number of outputs.
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
    input_dim
        The number of inputs.
    output_dim
        The number of outputs.
    tf
        The transfer function.
    dtf
        The complex derivative of the transfer function.
    """

    def __init__(self, input_space, output_space, tf, dtf, parameters={}, cont_time=True, name=None):
        super().__init__(input_space, output_space, cont_time=cont_time, name=name)
        self.parameters_own = parameters
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of inputs:  {self.input_dim}\n'
            f'    number of outputs: {self.output_dim}\n'
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
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) + other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: self.eval_dtf(s, mu=mu) + other.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    __radd__ = __add__

    def __sub__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) - other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: self.eval_dtf(s, mu=mu) - other.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    def __rsub__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

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
        assert self.input_space == other.output_space

        tf = lambda s, mu=None: self.eval_tf(s, mu=mu) @ other.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: self.eval_dtf(s, mu=mu) @ other.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

    def __rmul__(self, other):
        assert isinstance(other, InputOutputModel)
        assert self.cont_time == other.cont_time
        assert self.output_space == other.input_space

        tf = lambda s, mu=None: other.eval_tf(s, mu=mu) @ self.eval_tf(s, mu=mu)
        dtf = lambda s, mu=None: other.eval_dtf(s, mu=mu) @ self.eval_dtf(s, mu=mu)
        return self.with_(tf=tf, dtf=dtf)

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
        if 'epsabs' not in quad_kwargs:
            quad_kwargs['epsabs'] = 0
        quad_out = spint.quad(lambda w: spla.norm(self.eval_tf(w * 1j))**2,
                              -np.inf, np.inf,
                              **quad_kwargs)
        norm = np.sqrt(quad_out[0] / (2 * np.pi))
        if return_norm_only:
            return norm
        else:
            abserr = quad_out[1]
            norm_relerr = abserr / (2 * np.pi) / (2 * norm) / norm
            if len(quad_out) == 2:
                return norm, norm_relerr
            else:
                return norm, norm_relerr, quad_out[2:]


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
    estimator
        An error estimator for the problem. This can be any object with an `estimate(U, mu, model)`
        method. If `estimator` is not `None`, an `estimate(U, mu)` method is added to the model
        which will call `estimator.estimate(U, mu, self)`.
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
    input_dim
        The number of inputs.
    output_dim
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
                 solver_options=None, estimator=None, visualizer=None, name=None):

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

        super().__init__(B.source, M.source, Cp.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.input_dim}\n'
            f'    number of outputs:   {self.output_dim}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    second-order\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    @classmethod
    def from_matrices(cls, M, E, K, B, Cp, Cv=None, D=None, cont_time=True,
                      state_id='STATE', solver_options=None, estimator=None,
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
        estimator
            An error estimator for the problem. This can be any object with an
            `estimate(U, mu, model)` method. If `estimator` is not `None`, an `estimate(U, mu)`
            method is added to the model which will call `estimator.estimate(U, mu, self)`.
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
                   solver_options=solver_options, estimator=estimator, visualizer=visualizer, name=name)

    @classmethod
    def from_files(cls, M_file, E_file, K_file, B_file, Cp_file, Cv_file=None, D_file=None, cont_time=True,
                   state_id='STATE', solver_options=None, estimator=None, visualizer=None,
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
        estimator
            An error estimator for the problem. This can be any object with an
            `estimate(U, mu, model)` method. If `estimator` is not `None`, an `estimate(U, mu)`
            method is added to the model which will call `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with a `visualize(U, model, ...)`
            method. If `visualizer` is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the visualizer's `visualize` method.
        name
            Name of the system.

        Returns
        -------
        som
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
                                 estimator=estimator, visualizer=visualizer, name=name)

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
        return LTIModel(A=SecondOrderModelOperator(self.E, self.K),
                        B=BlockColumnOperator([ZeroOperator(self.B.range, self.B.source), self.B]),
                        C=BlockRowOperator([self.Cp, self.Cv]),
                        D=self.D,
                        E=(IdentityOperator(BlockVectorSpace([self.M.source, self.M.source]))
                           if isinstance(self.M, IdentityOperator) else
                           BlockDiagonalOperator([IdentityOperator(self.M.source), self.M])),
                        cont_time=self.cont_time,
                        solver_options=self.solver_options, estimator=self.estimator, visualizer=self.visualizer,
                        name=self.name + '_first_order')

    def __add__(self, other):
        """Add a |SecondOrderModel| or an |LTIModel|."""
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

        if isinstance(other, LTIModel):
            return self.to_lti() + other

        if not isinstance(other, SecondOrderModel):
            return NotImplemented

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
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

        if isinstance(other, LTIModel):
            return self.to_lti() - other

        if not isinstance(other, SecondOrderModel):
            return NotImplemented

        M = BlockDiagonalOperator([self.M, other.M])
        E = BlockDiagonalOperator([self.E, other.E])
        K = BlockDiagonalOperator([self.K, other.K])
        B = BlockColumnOperator([self.B, other.B])
        Cp = BlockRowOperator([self.Cp, -other.Cp])
        Cv = BlockRowOperator([self.Cv, -other.Cv])
        if self.D is other.D:
            D = ZeroOperator(self.output_space, self.input_space)
        else:
            D = self.D - other.D
        return self.with_(M=M, E=E, K=K, B=B, Cp=Cp, Cv=Cv, D=D)

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
        assert self.cont_time == other.cont_time
        assert self.input_space == other.output_space

        if isinstance(other, LTIModel):
            return self.to_lti() * other

        if not isinstance(other, SecondOrderModel):
            return NotImplemented

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
            Assumes that either the number of inputs or the number of outputs is much smaller than
            the order of the system.

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
            `(self.output_dim, self.input_dim)`.
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
        if self.input_dim <= self.output_dim:
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
            Assumes that either the number of inputs or the number of outputs is much smaller than
            the order of the system.

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
            shape `(self.output_dim, self.input_dim)`.
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
        if self.input_dim <= self.output_dim:
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
    estimator
        An error estimator for the problem. This can be any object with an `estimate(U, mu, model)`
        method. If `estimator` is not `None`, an `estimate(U, mu)` method is added to the model
        which will call `estimator.estimate(U, mu, self)`.
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
    input_dim
        The number of inputs.
    output_dim
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
                 estimator=None, visualizer=None, name=None):

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

        super().__init__(B.source, A.source, C.range, cont_time=cont_time,
                         estimator=estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.q = len(Ad)
        self.solution_space = A.source

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.input_dim}\n'
            f'    number of outputs:   {self.output_dim}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    time-delay\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )

    def __add__(self, other):
        """Add an |LTIModel|, |SecondOrderModel| or |LinearDelayModel|."""
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

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
        assert self.cont_time == other.cont_time
        assert self.input_space == other.input_space
        assert self.output_space == other.output_space

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

        E = BlockDiagonalOperator([self.E, other.E])
        A = BlockDiagonalOperator([self.A, other.A])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, -other.C])
        if self.D is other.D:
            D = ZeroOperator(self.output_space, self.input_space)
        else:
            D = self.D - other.D
        return self.with_(E=E, A=A, Ad=Ad, tau=tau, B=B, C=C, D=D)

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
        """Postmultiply by a |SecondOrderModel| or an |LTIModel|."""
        assert self.cont_time == other.cont_time
        assert self.input_space == other.output_space

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
        assert self.input_space == other.output_space

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
            Assumes that either the number of inputs or the number of outputs is much smaller than
            the order of the system.

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
            `(self.output_dim, self.input_dim)`.
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
        if self.input_dim <= self.output_dim:
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
            Assumes that either the number of inputs or the number of outputs is much smaller than
            the order of the system.

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
            shape `(self.output_dim, self.input_dim)`.
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
        if self.input_dim <= self.output_dim:
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
    estimator
        An error estimator for the problem. This can be any object with an `estimate(U, mu, model)`
        method. If `estimator` is not `None`, an `estimate(U, mu)` method is added to the model
        which will call `estimator.estimate(U, mu, self)`.
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
    input_dim
        The number of inputs.
    output_dim
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
                 estimator=None, visualizer=None, name=None):

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
                         estimator=estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.q = len(As)

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.input_dim}\n'
            f'    number of outputs:   {self.output_dim}\n'
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
    estimator
        An error estimator for the problem. This can be any object with an `estimate(U, mu, model)`
        method. If `estimator` is not `None`, an `estimate(U, mu)` method is added to the model
        which will call `estimator.estimate(U, mu, self)`.
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
    input_dim
        The number of inputs.
    output_dim
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
                 estimator=None, visualizer=None, name=None):

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
                         estimator=estimator, visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.linear = False

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.input_dim}\n'
            f'    number of outputs:   {self.output_dim}\n'
            f'    {"continuous" if self.cont_time else "discrete"}-time\n'
            f'    bilinear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
