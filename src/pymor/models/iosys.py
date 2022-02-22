# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.bernoulli import bernoulli_stabilize
from pymor.algorithms.eigs import eigs
from pymor.algorithms.lyapunov import solve_lyap_lrcf, solve_lyap_dense
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import cached
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.models.interface import Model
from pymor.models.transfer_function import FactorizedTransferFunction
from pymor.operators.block import (BlockOperator, BlockRowOperator, BlockColumnOperator, BlockDiagonalOperator,
                                   SecondOrderModelOperator)
from pymor.operators.constructions import IdentityOperator, LincombOperator, LowRankOperator, ZeroOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameters, Mu
from pymor.vectorarrays.block import block


@defaults('value')
def sparse_min_size(value=1000):
    """Return minimal sparse problem size for which to warn about converting to dense."""
    return value


class LTIModel(Model):
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

    All methods related to the transfer function
    (e.g., frequency response calculation and Bode plots)
    are attached to the `transfer_function` attribute.

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
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
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
    transfer_function
        The transfer function.
    """

    def __init__(self, A, B, C, D=None, E=None, sampling_time=0,
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

        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        assert solver_options is None or solver_options.keys() <= {'lyap_lrcf', 'lyap_dense'}

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = A.source
        self.dim_output = C.range.dim

        K = lambda s: s * self.E - self.A
        B = lambda s: self.B
        C = lambda s: self.C
        D = lambda s: self.D
        dK = lambda s: self.E
        dB = lambda s: ZeroOperator(self.B.range, self.B.source)
        dC = lambda s: ZeroOperator(self.C.range, self.C.source)
        dD = lambda s: ZeroOperator(self.D.range, self.D.source)
        parameters = Parameters.of(self.A, self.B, self.C, self.D, self.E)

        self.transfer_function = FactorizedTransferFunction(
            self.dim_input, self.dim_output,
            K, B, C, D, dK, dB, dC, dD,
            parameters=parameters, sampling_time=sampling_time, name=self.name + '_transfer_function')

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time\n'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}\n'
        string += (
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
        return string

    @classmethod
    def from_matrices(cls, A, B, C, D=None, E=None, sampling_time=0,
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
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
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

        return cls(A, B, C, D, E, sampling_time=sampling_time,
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
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, sampling_time=0,
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
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
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

        return cls.from_matrices(A, B, C, D, E, sampling_time=sampling_time,
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
    def from_mat_file(cls, file_name, sampling_time=0,
                      state_id='STATE', solver_options=None, error_estimator=None,
                      visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in a .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be included) containing A, B,
            C, and optionally D and E.
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
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

        return cls.from_matrices(A, B, C, D, E, sampling_time=sampling_time,
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
    def from_abcde_files(cls, files_basename, sampling_time=0,
                         state_id='STATE', solver_options=None, error_estimator=None,
                         visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in .[ABCDE] files.

        Parameters
        ----------
        files_basename
            The basename of files containing A, B, C, and optionally D and E.
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
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

        return cls.from_matrices(A, B, C, D, E, sampling_time=sampling_time,
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

        assert self.sampling_time == other.sampling_time
        assert self.D.source == other.D.source
        assert self.D.range == other.D.range

        A = BlockDiagonalOperator([self.A, other.A])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, other.C])
        D = self.D + other.D
        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(block([self.solution_space, other.solution_space]))
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

        assert self.sampling_time == other.sampling_time
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
        if self.sampling_time > 0:
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
        if self.sampling_time > 0:
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
        dico = 'D' if self.sampling_time > 0 else 'C'
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


class SecondOrderModel(Model):
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

    All methods related to the transfer function
    (e.g., frequency response calculation and Bode plots)
    are attached to the `transfer_function` attribute.

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
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
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
    transfer_function
        The transfer function.
    """

    def __init__(self, M, E, K, B, Cp, Cv=None, D=None, sampling_time=0,
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

        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        assert solver_options is None or solver_options.keys() <= {'lyap_lrcf', 'lyap_dense'}

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = M.source
        self.dim_output = Cp.range.dim

        K = lambda s: s**2 * self.M + s * self.E + self.K
        B = lambda s: self.B
        C = lambda s: self.Cp + s * self.Cv
        D = lambda s: self.D
        dK = lambda s: 2 * s * self.M + self.E
        dB = lambda s: ZeroOperator(self.B.range, self.B.source)
        dC = lambda s: self.Cv
        dD = lambda s: ZeroOperator(self.D.range, self.D.source)
        parameters = Parameters.of(self.M, self.E, self.K, self.B, self.Cp, self.Cv, self.D)

        self.transfer_function = FactorizedTransferFunction(
            self.dim_input, self.dim_output,
            K, B, C, D, dK, dB, dC, dD,
            parameters=parameters, sampling_time=sampling_time, name=self.name + '_transfer_function')

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time\n'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}\n'
        string += (
            f'    second-order\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
        return string

    @classmethod
    def from_matrices(cls, M, E, K, B, Cp, Cv=None, D=None, sampling_time=0,
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
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
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

        return cls(M, E, K, B, Cp, Cv, D, sampling_time=sampling_time,
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
    def from_files(cls, M_file, E_file, K_file, B_file, Cp_file, Cv_file=None, D_file=None, sampling_time=0,
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
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
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

        return cls.from_matrices(M, E, K, B, Cp, Cv, D, sampling_time=sampling_time,
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
                        E=(IdentityOperator(block([self.M.source, self.M.source]))
                           if isinstance(self.M, IdentityOperator) else
                           BlockDiagonalOperator([IdentityOperator(self.M.source), self.M])),
                        sampling_time=self.sampling_time,
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

        assert self.sampling_time == other.sampling_time
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

        assert self.sampling_time == other.sampling_time
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


class LinearDelayModel(Model):
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

    All methods related to the transfer function
    (e.g., frequency response calculation and Bode plots)
    are attached to the `transfer_function` attribute.

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
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
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
    transfer_function
        The transfer function.
    """

    def __init__(self, A, Ad, tau, B, C, D=None, E=None, sampling_time=0,
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

        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = A.source
        self.dim_output = C.range.dim
        self.q = len(Ad)

        K = lambda s: LincombOperator((E, A) + Ad, (s, -1) + tuple(-np.exp(-taui * s) for taui in self.tau))
        B = lambda s: self.B
        C = lambda s: self.C
        D = lambda s: self.D
        dK = lambda s: LincombOperator((E,) + Ad, (1,) + tuple(taui * np.exp(-taui * s) for taui in self.tau))
        dB = lambda s: ZeroOperator(self.B.range, self.B.source)
        dC = lambda s: ZeroOperator(self.C.range, self.C.source)
        dD = lambda s: ZeroOperator(self.D.range, self.D.source)
        parameters = Parameters.of(self.A, self.Ad,  self.B, self.C, self.D, self.E)

        self.transfer_function = FactorizedTransferFunction(
            self.dim_input, self.dim_output,
            K, B, C, D, dK, dB, dC, dD,
            parameters=parameters, sampling_time=sampling_time, name=self.name + '_transfer_function')

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time\n'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}\n'
        string += (
            f'    time-delay\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
        return string

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

        assert self.sampling_time == other.sampling_time
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

        assert self.sampling_time == other.sampling_time
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
        assert self.sampling_time == other.sampling_time
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


class LinearStochasticModel(Model):
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
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
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

    def __init__(self, A, As, B, C, D=None, E=None, sampling_time=0,
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

        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = A.source
        self.dim_output = C.range.dim
        self.q = len(As)

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time\n'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}\n'
        string += (
            f'    stochastic\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
        return string


class BilinearModel(Model):
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
    sampling_time
        `0` if the system is continuous-time, otherwise a positive number that denotes the
        sampling time (in seconds).
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

    def __init__(self, A, N, B, C, D, E=None, sampling_time=0,
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

        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = A.source
        self.dim_output = C.range.dim
        self.linear = False

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
        )
        if self.sampling_time == 0:
            string += '    continuous-time\n'
        else:
            string += f'    {f"discrete-time (sampling_time={self.sampling_time:.2e}s)"}\n'
        string += (
            f'    bilinear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
        return string


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
