# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
from numbers import Number

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sps

from pymor.algorithms.bernoulli import bernoulli_stabilize
from pymor.algorithms.eigs import eigs
from pymor.algorithms.lyapunov import (
    _chol,
    solve_cont_lyap_dense,
    solve_cont_lyap_lrcf,
    solve_disc_lyap_dense,
    solve_disc_lyap_lrcf,
)
from pymor.algorithms.riccati import solve_pos_ricc_dense, solve_pos_ricc_lrcf, solve_ricc_lrcf
from pymor.algorithms.simplify import contract, expand
from pymor.algorithms.timestepping import DiscreteTimeStepper, TimeStepper
from pymor.algorithms.to_matrix import to_matrix
from pymor.analyticalproblems.functions import Function
from pymor.core.cache import cached
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.models.interface import Model
from pymor.models.transfer_function import FactorizedTransferFunction
from pymor.models.transforms import BilinearTransformation, MoebiusTransformation
from pymor.operators.block import (
    BlockColumnOperator,
    BlockDiagonalOperator,
    BlockOperator,
    BlockRowOperator,
    SecondOrderModelOperator,
)
from pymor.operators.constructions import (
    IdentityOperator,
    LincombOperator,
    LinearInputOperator,
    LowRankOperator,
    VectorArrayOperator,
    VectorOperator,
    ZeroOperator,
)
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Mu, Parameters
from pymor.parameters.functionals import ExpressionParameterFunctional, ProjectionParameterFunctional
from pymor.vectorarrays.block import BlockVectorSpace
from pymor.vectorarrays.interface import VectorArray


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
    T
        The final time T.
    initial_data
        The initial data `x_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for given |parameter values|. If `None`, it is
        assumed to be zero.
    time_stepper
        The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
        to be used by :meth:`~pymor.models.interface.Model.solve`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    presets
        A `dict` of preset attributes or `None`. The dict must only contain keys that correspond to
        attributes of |LTIModel| such as `poles`, `c_lrcf`, `o_lrcf`, `c_dense`, `o_dense`, `hsv`,
        `h2_norm`, `hinf_norm`, `l2_norm` and `linf_norm`. Additionally, the frequency at which the
        :math:`\mathcal{H}_\infty/\mathcal{L}_\infty` norm is attained can be preset with `fpeak`.
    solver_options
        The solver options to use to solve matrix equations.
    ast_pole_data
        Used in :meth:`get_ast_spectrum`. Can be:

        - dictionary of parameters for :func:`~pymor.algorithms.eigs.eigs`,
        - list of anti-stable eigenvalues (scalars),
        - tuple `(lev, ew, rev)` where `ew` contains the sorted anti-stable eigenvalues
            and `lev` and `rev` are |VectorArrays| representing the eigenvectors.
        - `None` if anti-stable eigenvalues should be computed via dense methods.
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

    cache_region = 'memory'

    def __init__(self, A, B, C, D=None, E=None, sampling_time=0,
                 T=None, initial_data=None, time_stepper=None, num_values=None, presets=None,
                 solver_options=None, ast_pole_data=None,
                 error_estimator=None, visualizer=None, name=None):

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

        assert T is None or T > 0

        if T is not None:
            if initial_data is None:
                initial_data = A.source.zeros(1)
            if isinstance(initial_data, VectorArray):
                assert initial_data in A.source
                assert len(initial_data) == 1
                initial_data = VectorOperator(initial_data, name='initial_data')
            assert initial_data.source.is_scalar
            assert initial_data.range == A.source

            if sampling_time == 0:
                assert isinstance(time_stepper, TimeStepper)
                assert not isinstance(time_stepper, DiscreteTimeStepper)
            else:
                if time_stepper is None:
                    time_stepper = DiscreteTimeStepper()
                assert isinstance(time_stepper, DiscreteTimeStepper)
        else:
            if initial_data is not None:
                raise ValueError('Initial data is given but T is not.')
            if time_stepper is not None:
                raise ValueError('Time-stepper is given but T is not.')

        assert presets is None or presets.keys() <= {'poles', 'c_lrcf', 'o_lrcf', 'c_dense', 'o_dense', 'hsv',
                                                     'h2_norm', 'hinf_norm', 'l2_norm', 'linf_norm', 'fpeak'}
        if presets:
            assert all(not obj.parametric for obj in [A, B, C, D, E])
        else:
            presets = {}

        assert solver_options is None or solver_options.keys() <= {'lyap_lrcf', 'lyap_dense',
                                                                   'ricc_lrcf', 'ricc_dense',
                                                                   'ricc_pos_dense', 'ricc_pos_lrcf'}

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = A.source
        self.dim_output = C.range.dim

        parameters = Parameters.of(self.A, self.B, self.C, self.D, self.E)
        s = ProjectionParameterFunctional('s')

        K = s * self.E - self.A
        B = self.B
        C = self.C
        D = self.D
        dK = self.E
        dB = ZeroOperator(self.B.range, self.B.source)
        dC = ZeroOperator(self.C.range, self.C.source)
        dD = ZeroOperator(self.D.range, self.D.source)

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
                      T=None, initial_data=None, time_stepper=None, num_values=None, presets=None,
                      state_id=None, solver_options=None, error_estimator=None,
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
        T
            The final time T.
        initial_data
            The initial data `x_0` as a |NumPy array|. If `None`, it is assumed
            to be zero.
        time_stepper
            The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
            to be used by :meth:`~pymor.models.interface.Model.solve`.
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.
        presets
            A `dict` of preset attributes or `None`.
            See |LTIModel|.
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
        assert isinstance(A, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(B, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(C, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(D, (np.ndarray, sps.spmatrix, sps.sparray, type(None)))
        assert isinstance(E, (np.ndarray, sps.spmatrix, sps.sparray, type(None)))
        assert isinstance(initial_data, (np.ndarray, type(None)))

        A = NumpyMatrixOperator(A, source_id=state_id, range_id=state_id)
        B = NumpyMatrixOperator(B, range_id=state_id)
        C = NumpyMatrixOperator(C, source_id=state_id)
        if D is not None:
            D = NumpyMatrixOperator(D)
        if E is not None:
            E = NumpyMatrixOperator(E, source_id=state_id, range_id=state_id)
        if initial_data is not None:
            initial_data = A.source.from_numpy(initial_data)

        return cls(A, B, C, D, E, sampling_time=sampling_time, T=T, initial_data=initial_data,
                   time_stepper=time_stepper, num_values=num_values, presets=presets,
                   solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer, name=name)

    def to_matrices(self, format=None, mu=None):
        """Return operators as matrices.

        Parameters
        ----------
        format
            Format of the resulting matrices: |NumPy array| if 'dense',
            otherwise the appropriate |SciPy spmatrix|.
            If `None`, a choice between dense and sparse format is
            automatically made.
        mu
            The |parameter values| for which to convert the operators.

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
        return self.to_abcde_matrices(format, mu)

    def to_abcde_matrices(self, format=None, mu=None):
        """Return A, B, C, D, and E operators as matrices.

        Parameters
        ----------
        format
            Format of the resulting matrices: |NumPy array| if 'dense',
            otherwise the appropriate |SciPy spmatrix|.
            If `None`, a choice between dense and sparse format is
            automatically made.
        mu
            The |parameter values| for which to convert the operators.

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
        A = to_matrix(self.A, format, mu)
        B = to_matrix(self.B, format, mu)
        C = to_matrix(self.C, format, mu)
        D = None if isinstance(self.D, ZeroOperator) else to_matrix(self.D, format, mu)
        E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E, format, mu)
        return A, B, C, D, E

    @classmethod
    def from_files(cls, A_file, B_file, C_file, D_file=None, E_file=None, sampling_time=0,
                   T=None, initial_data_file=None, time_stepper=None, num_values=None, presets=None,
                   state_id=None, solver_options=None, error_estimator=None, visualizer=None, name=None):
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
        T
            The final time T.
        initial_data_file
            `None` or the name of the file (with extension) containing the
            initial data.
        time_stepper
            The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
            to be used by :meth:`~pymor.models.interface.Model.solve`.
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.
        presets
            A `dict` of preset attributes or `None`.
            See |LTIModel|.
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
        initial_data = load_matrix(initial_data_file) if initial_data_file is not None else None

        return cls.from_matrices(A, B, C, D, E, sampling_time=sampling_time, T=T, initial_data=initial_data,
                                 time_stepper=time_stepper, num_values=num_values, presets=presets, state_id=state_id,
                                 solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer,
                                 name=name)

    def to_files(self, A_file, B_file, C_file, D_file=None, E_file=None, mu=None):
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
        mu
            The |parameter values| for which to write the operators to files.
        """
        if D_file is None and not isinstance(self.D, ZeroOperator):
            raise ValueError('D is not zero, D_file must be given')
        if E_file is None and not isinstance(self.E, IdentityOperator):
            raise ValueError('E is not identity, E_file must be given')

        from pymor.tools.io import save_matrix

        A, B, C, D, E = self.to_matrices(mu=mu)
        for mat, file in [(A, A_file), (B, B_file), (C, C_file), (D, D_file), (E, E_file)]:
            if mat is None:
                continue
            save_matrix(file, mat)

    @classmethod
    def from_mat_file(cls, file_name, sampling_time=0, T=None, time_stepper=None, num_values=None, presets=None,
                      state_id=None, solver_options=None, error_estimator=None, visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in a .mat file.

        Supports the format used in the `SLICOT benchmark collection
        <http://slicot.org/20-site/126-benchmark-examples-for-model-reduction>`_.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be included) containing A, B,
            and optionally C, D, and E.
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
        T
            The final time T.
        time_stepper
            The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
            to be used by :meth:`~pymor.models.interface.Model.solve`.
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.
        presets
            A `dict` of preset attributes or `None`.
            See |LTIModel|.
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

        assert 'A' in mat_dict
        assert 'B' in mat_dict

        matrices = [
            mat_dict['A'],
            mat_dict['B'],
            mat_dict.get('C', mat_dict['B'].T),
            mat_dict.get('D'),
            mat_dict.get('E'),
        ]

        # convert integer dtypes to floating dtypes
        for i in range(len(matrices)):
            mat = matrices[i]
            if mat is not None and np.issubdtype(mat.dtype, np.integer):
                matrices[i] = mat.astype(np.float64)

        return cls.from_matrices(*matrices, sampling_time=sampling_time, T=T, time_stepper=time_stepper,
                                 num_values=num_values, presets=presets, state_id=state_id,
                                 solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer,
                                 name=name)

    def to_mat_file(self, file_name, mu=None):
        """Save operators as matrices to .mat file.

        Parameters
        ----------
        file_name
            The name of the .mat file (extension .mat does not need to be included).
        mu
            The |parameter values| for which to write the operators to files.
        """
        import scipy.io as spio
        A, B, C, D, E = self.to_matrices(mu=mu)
        mat_dict = {'A': A, 'B': B, 'C': C}
        if D is not None:
            mat_dict['D'] = D
        if E is not None:
            mat_dict['E'] = E
        spio.savemat(file_name, mat_dict)

    @classmethod
    def from_abcde_files(cls, files_basename, sampling_time=0, T=None, time_stepper=None, num_values=None, presets=None,
                         state_id=None, solver_options=None, error_estimator=None, visualizer=None, name=None):
        """Create |LTIModel| from matrices stored in .[ABCDE] files.

        Parameters
        ----------
        files_basename
            The basename of files containing A, B, C, and optionally D and E.
        sampling_time
            `0` if the system is continuous-time, otherwise a positive number that denotes the
            sampling time (in seconds).
        T
            The final time T.
        time_stepper
            The :class:`time-stepper <pymor.algorithms.timestepping.TimeStepper>`
            to be used by :meth:`~pymor.models.interface.Model.solve`.
        num_values
            The number of returned vectors of the solution trajectory. If `None`, each
            intermediate vector that is calculated is returned.
        presets
            A `dict` of preset attributes or `None`.
            See |LTIModel|.
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
        import os.path

        from pymor.tools.io import load_matrix

        A = load_matrix(files_basename + '.A')
        B = load_matrix(files_basename + '.B')
        C = load_matrix(files_basename + '.C')
        D = load_matrix(files_basename + '.D') if os.path.isfile(files_basename + '.D') else None
        E = load_matrix(files_basename + '.E') if os.path.isfile(files_basename + '.E') else None

        return cls.from_matrices(A, B, C, D, E, sampling_time=sampling_time, T=T, time_stepper=time_stepper,
                                 num_values=num_values, presets=presets, state_id=state_id,
                                 solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer,
                                 name=name)

    def to_abcde_files(self, files_basename, mu=None):
        """Save operators as matrices to .[ABCDE] files in Matrix Market format.

        Parameters
        ----------
        files_basename
            The basename of files containing the operators.
        mu
            The |parameter values| for which to write the operators to files.
        """
        from pathlib import Path

        from pymor.tools.io.matrices import _mmwrite
        A, B, C, D, E = self.to_matrices(mu=mu)
        _mmwrite(Path(files_basename + '.A'), A)
        _mmwrite(Path(files_basename + '.B'), B)
        _mmwrite(Path(files_basename + '.C'), C)
        if D is not None:
            _mmwrite(Path(files_basename + '.D'), D)
        if E is not None:
            _mmwrite(Path(files_basename + '.E'), E)

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities or 'output' in quantities:
            assert self.T is not None

            compute_solution = 'solution' in quantities
            compute_output = 'output' in quantities

            # solution computation
            iterator = self.time_stepper.iterate(
                0,  # initial_time
                self.T,  # end_time
                self.initial_data.as_range_array(mu),  # initial_data
                -self.A,  # operator
                rhs=LinearInputOperator(self.B),
                mass=None if isinstance(self.E, IdentityOperator) else self.E,
                mu=mu.with_(t=0),
                num_values=self.num_values
            )
            if self.num_values is None:
                try:
                    n = self.time_stepper.estimate_time_step_count(0, self.T) + 1
                except NotImplementedError:
                    n = 0
            else:
                n = self.num_values + 1

            if compute_solution:
                data['solution'] = self.solution_space.empty(reserve=n)
            if compute_output:
                D = LinearInputOperator(self.D)
                data['output'] = np.empty((n, self.dim_output))
                data_output_extra = []
            for i, (x, t) in enumerate(iterator):
                if compute_solution:
                    data['solution'].append(x)
                if compute_output:
                    y = self.C.apply(x, mu=mu).to_numpy() + D.as_range_array(mu=mu.with_(t=t)).to_numpy()
                    if i < n:
                        data['output'][i] = y
                    else:
                        data_output_extra.append(y)
            if compute_output:
                if data_output_extra:
                    data['output'] = np.vstack((data['output'], data_output_extra))
                if len(data['output']) < i + 1:
                    data['output'] = data['output'][:i + 1]

            if compute_solution:
                quantities.remove('solution')
            if compute_output:
                quantities.remove('output')

        super()._compute(quantities, data, mu=mu)

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
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            E = BlockDiagonalOperator([self.E, other.E])
        if self.T is not None and other.T is not None:
            if type(self.time_stepper) != type(other.time_stepper):  # noqa: E721
                raise TypeError('The time-steppers are not of the same type.')
            T = min(self.T, other.T)
            initial_data = BlockColumnOperator([self.initial_data, other.initial_data])
            time_stepper = self.time_stepper
            if (hasattr(self.time_stepper, 'nt') and hasattr(other.time_stepper, 'nt')
                    and self.T / self.time_stepper.nt > other.T / other.time_stepper.nt):
                time_stepper = other.time_stepper
        else:
            T = None
            initial_data = None
            time_stepper = None
        return LTIModel(A, B, C, D, E, sampling_time=self.sampling_time,
                        T=T, initial_data=initial_data, time_stepper=time_stepper, num_values=self.num_values,
                        solver_options=self.solver_options)

    def __sub__(self, other):
        """Subtract an |LTIModel|."""
        return self + (-other)

    def __neg__(self):
        """Negate the |LTIModel|."""
        return self.with_(C=-self.C, D=-self.D, new_type=LTIModel)  # ensure that __neg__ works in subclasses

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
        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            E = BlockDiagonalOperator([self.E, other.E])
        if self.T is not None and other.T is not None:
            if type(self.time_stepper) != type(other.time_stepper):  # noqa: E721
                raise TypeError('The time-steppers are not of the same type.')
            T = min(self.T, other.T)
            initial_data = BlockColumnOperator([self.initial_data, other.initial_data])
            time_stepper = self.time_stepper
            if (hasattr(self.time_stepper, 'nt') and hasattr(other.time_stepper, 'nt')
                    and self.T / self.time_stepper.nt > other.T / other.time_stepper.nt):
                time_stepper = other.time_stepper
        else:
            T = None
            initial_data = None
            time_stepper = None
        return LTIModel(A, B, C, D, E, sampling_time=self.sampling_time,
                        T=T, initial_data=initial_data, time_stepper=time_stepper, num_values=self.num_values,
                        solver_options=self.solver_options)

    def impulse_resp(self, mu=None, return_solution=False):
        """Compute impulse response from all inputs.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.
        return_solution
            If `True`, the model :meth:`solution <pymor.models.interface.Model.solve>` for the given
            |parameter values| `mu` is returned.

        Returns
        -------
        output
            Impulse response as a 3D |NumPy array| where
            `output.shape[0]` is the number of time steps,
            `output.shape[1]` is the number of outputs, and
            `output.shape[2]` is the number of inputs.
        solution
            The tuple of solution |VectorArrays| for every input.
            Returned only when `return_solution` is `True`.
        """
        assert self.T is not None

        if self.num_values is None:
            try:
                n = self.time_stepper.estimate_time_step_count(0, self.T) + 1
            except NotImplementedError:
                n = 0
        else:
            n = self.num_values + 1
        output = np.empty((n, self.dim_output, self.dim_input))
        if return_solution:
            solution = []

        if self.sampling_time == 0:
            initial_data = self.E.apply_inverse(self.B.as_range_array(mu=mu), mu=mu)

        for i in range(self.dim_input):
            if self.sampling_time == 0:
                data = self.with_(initial_data=initial_data[i]).compute(
                    input=np.zeros(self.dim_input), output=True, solution=return_solution, mu=mu)
            else:
                input = ImpulseFunction(self.dim_input, i, self.sampling_time)
                data = self.with_(initial_data=self.solution_space.zeros(1)).compute(
                    input=input, output=True, solution=return_solution, mu=mu)

            output[..., i] = data['output']
            if return_solution:
                solution.append(data['solution'])

        if return_solution:
            return output, tuple(solution)

        return output

    def step_resp(self, mu=None, return_solution=False):
        """Compute step response from all inputs.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.
        return_solution
            If `True`, the model solution for the given |parameter values| `mu` is returned.

        Returns
        -------
        output
            Step response as a 3D |NumPy array| where
            `output.shape[0]` is the number of time steps,
            `output.shape[1]` is the number of outputs, and
            `output.shape[2]` is the number of inputs.
        solution
            The tuple of solution |VectorArrays| for every input.
            Returned only when `return_solution` is `True`.
        """
        assert self.T is not None

        if self.num_values is None:
            try:
                n = self.time_stepper.estimate_time_step_count(0, self.T) + 1
            except NotImplementedError:
                n = 0
        else:
            n = self.num_values + 1
        output = np.empty((n, self.dim_output, self.dim_input))
        if return_solution:
            solution = []

        for i in range(self.dim_input):
            input = StepFunction(self.dim_input, i, self.sampling_time)
            data = self.compute(input=input, output=True, solution=return_solution, mu=mu)
            output[..., i] = data['output']
            if return_solution:
                solution.append(data['solution'])

        if return_solution:
            return output, tuple(solution)

        return output

    @cached
    def _poles(self, mu=None):
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
        poles = self.presets['poles'] if 'poles' in self.presets else self._poles(mu=mu)
        assert isinstance(poles, np.ndarray)
        assert poles.shape == (self.A.source.dim,)

        return poles

    @cached
    def _gramian(self, typ, mu=None):
        if typ == 'c_lrcf' and 'c_dense' in self.presets:
            return self.A.source.from_numpy(_chol(self.presets['c_dense']).T)
        elif typ == 'o_lrcf' and 'o_dense' in self.presets:
            return self.A.source.from_numpy(_chol(self.presets['o_dense']).T)
        elif typ == 'c_dense' and 'c_lrcf' in self.presets:
            return self.presets['c_lrcf'].to_numpy().T @ self.presets['c_lrcf'].to_numpy()
        elif typ == 'o_dense' and 'o_lrcf' in self.presets:
            return self.presets['o_lrcf'].to_numpy().T @ self.presets['o_lrcf'].to_numpy()

        A = self.A.assemble(mu)
        B = self.B
        C = self.C
        D = self.D
        E = self.E.assemble(mu) if not isinstance(self.E, IdentityOperator) else None
        options_lrcf = self.solver_options.get('lyap_lrcf') if self.solver_options else None
        options_dense = self.solver_options.get('lyap_dense') if self.solver_options else None
        options_ricc_lrcf = self.solver_options.get('ricc_lrcf') if self.solver_options else None
        options_ricc_pos_dense = self.solver_options.get('ricc_pos_dense') if self.solver_options else None
        options_ricc_pos_lrcf = self.solver_options.get('ricc_pos_lrcf') if self.solver_options else None
        solve_lyap_lrcf = solve_cont_lyap_lrcf if self.sampling_time == 0 else solve_disc_lyap_lrcf
        solve_lyap_dense = solve_cont_lyap_dense if self.sampling_time == 0 else solve_disc_lyap_dense

        if typ == 'c_lrcf':
            return solve_lyap_lrcf(A, E, B.as_range_array(mu=mu), trans=False, options=options_lrcf)
        elif typ == 'o_lrcf':
            return solve_lyap_lrcf(A, E, C.as_source_array(mu=mu), trans=True, options=options_lrcf)
        elif typ == 'c_dense':
            return solve_lyap_dense(to_matrix(A, format='dense'), to_matrix(E, format='dense') if E else None,
                                    to_matrix(B, format='dense', mu=mu), trans=False, options=options_dense)
        elif typ == 'o_dense':
            return solve_lyap_dense(to_matrix(A, format='dense'), to_matrix(E, format='dense') if E else None,
                                    to_matrix(C, format='dense', mu=mu), trans=True, options=options_dense)
        elif typ == 'bs_c_lrcf':
            ast_spectrum = self.get_ast_spectrum(mu=mu)
            K = bernoulli_stabilize(A, E, B.as_range_array(mu=mu), ast_spectrum, trans=True)
            BK = LowRankOperator(B.as_range_array(mu=mu), np.eye(len(K)), K)
            return solve_cont_lyap_lrcf(A - BK, E, B.as_range_array(mu=mu),
                                        trans=False, options=options_lrcf)
        elif typ == 'bs_o_lrcf':
            ast_spectrum = self.get_ast_spectrum(mu=mu)
            K = bernoulli_stabilize(A, E, C.as_source_array(mu=mu), ast_spectrum, trans=False)
            KC = LowRankOperator(K, np.eye(len(K)), C.as_source_array(mu=mu))
            return solve_cont_lyap_lrcf(A - KC, E, C.as_source_array(mu=mu),
                                        trans=True, options=options_lrcf)
        elif typ == 'lqg_c_lrcf':
            return solve_ricc_lrcf(A, E, B.as_range_array(mu=mu), C.as_source_array(mu=mu),
                                   trans=False, options=options_ricc_lrcf)
        elif typ == 'lqg_o_lrcf':
            return solve_ricc_lrcf(A, E, B.as_range_array(mu=mu), C.as_source_array(mu=mu),
                                   trans=True, options=options_ricc_lrcf)
        elif typ == 'pr_c_lrcf':
            return solve_pos_ricc_lrcf(A, E, A.source.zeros(), -C.as_source_array(mu=mu),
                                       R=to_matrix(D + D.H, 'dense'), S=B.as_range_array(mu=mu),
                                       trans=False, options=options_ricc_pos_lrcf)
        elif typ == 'pr_o_lrcf':
            return solve_pos_ricc_lrcf(A, E, -B.as_range_array(mu=mu), A.source.zeros(),
                                       R=to_matrix(D + D.H, 'dense'), S=C.as_source_array(mu=mu),
                                       trans=True, options=options_ricc_pos_lrcf)
        elif typ == 'pr_c_dense':
            return solve_pos_ricc_dense(to_matrix(A, format='dense'), to_matrix(E, format='dense') if E else None,
                                        A.source.zeros().to_numpy().T, -to_matrix(C, format='dense'),
                                        R=to_matrix(D + D.H, 'dense'), S=to_matrix(B, format='dense').T,
                                        trans=False, options=options_ricc_pos_dense)
        elif typ == 'pr_o_dense':
            return solve_pos_ricc_dense(to_matrix(A, format='dense'), to_matrix(E, format='dense') if E else None,
                                        -to_matrix(B, format='dense'), A.source.zeros().to_numpy(),
                                        R=to_matrix(D + D.H, 'dense'), S=to_matrix(C, format='dense').T,
                                        trans=True, options=options_ricc_pos_dense)
        elif typ[0] == 'br_c_lrcf':
            return solve_pos_ricc_lrcf(A, E, B.as_range_array(mu=mu), C.as_source_array(mu=mu),
                                       R=(typ[1] ** 2 * np.eye(self.dim_output)
                                          if typ[1] != 1
                                          else None),
                                       trans=False, options=options_ricc_pos_lrcf)
        elif typ[0] == 'br_o_lrcf':
            return solve_pos_ricc_lrcf(A, E, B.as_range_array(mu=mu), C.as_source_array(mu=mu),
                                       R=(typ[1] ** 2 * np.eye(self.dim_input)
                                          if typ[1] != 1
                                          else None),
                                       trans=True, options=options_ricc_pos_lrcf)

    def gramian(self, typ, mu=None):
        """Compute a Gramian.

        Parameters
        ----------
        typ
            The type of the Gramian:

            - `'c_lrcf'`: low-rank Cholesky factor of the controllability Gramian,
            - `'o_lrcf'`: low-rank Cholesky factor of the observability Gramian,
            - `'c_dense'`: dense controllability Gramian,
            - `'o_dense'`: dense observability Gramian,
            - `'bs_c_lrcf'`: low-rank Cholesky factor of the Bernoulli stabilized controllability
              Gramian,
            - `'bs_o_lrcf'`: low-rank Cholesky factor of the Bernoulli stabilized observability
              Gramian,
            - `'lqg_c_lrcf'`: low-rank Cholesky factor of the "controllability" LQG Gramian,
            - `'lqg_o_lrcf'`: low-rank Cholesky factor of the "observability" LQG Gramian,
            - `('br_c_lrcf', gamma)`: low-rank Cholesky factor of the "controllability" bounded real
              Gramian,
            - `('br_o_lrcf', gamma)`: low-rank Cholesky factor of the "observability" bounded real
              Gramian.
            - `'pr_c_lrcf'`: low-rank Cholesky factor of the "controllability" positive real
              Gramian,
            - `'pr_o_lrcf'`: low-rank Cholesky factor of the "observability" positive real
              Gramian.

            .. note::
                For `'*_lrcf'` types, the method assumes the system is asymptotically stable.
                For `'*_dense'` types, the method assumes that the underlying Lyapunov equation
                has a unique solution, i.e. no pair of system poles adds to zero in the
                continuous-time case and no pair of system poles multiplies to one in the
                discrete-time case.
                Additionally, for `'pr_c_lrcf'` and `'pr_o_lrcf'`, it is assumed that `D + D^T` is
                invertible.
        mu
            |Parameter values|.

        Returns
        -------
        If typ ends with `'_lrcf'`, then the Gramian factor as a |VectorArray| from `self.A.source`.
        If typ ends with `'_dense'`, then the Gramian as a |NumPy array|.
        """
        assert (typ in ('c_lrcf', 'o_lrcf', 'c_dense', 'o_dense', 'bs_c_lrcf', 'bs_o_lrcf', 'lqg_c_lrcf', 'lqg_o_lrcf',
                        'pr_c_lrcf', 'pr_o_lrcf', 'pr_c_dense', 'pr_o_dense')
                or isinstance(typ, tuple) and len(typ) == 2 and typ[0] in ('br_c_lrcf', 'br_o_lrcf'))

        if ((isinstance(typ, str) and (typ.startswith('bs') or typ.startswith('lqg')) or isinstance(typ, tuple))
                and self.sampling_time > 0):
            raise NotImplementedError

        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)

        gramian = self.presets[typ] if typ in self.presets else self._gramian(typ, mu=mu)

        # assert correct return types
        assert ((isinstance(typ, str) and typ.endswith('_lrcf') or isinstance(typ, tuple))
                and gramian in self.A.source
                or isinstance(gramian, np.ndarray)
                and gramian.shape == (self.A.source.dim, self.A.source.dim))

        return gramian

    @cached
    def _sv_U_V(self, typ='lyap', mu=None):
        """Compute (Hankel) singular values and vectors.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        typ
            The type of the Gramians used:

            - `'lyap'`: Lyapunov Gramian,
            - `'bs'`: Bernoulli stabilized Gramian,
            - `'lqg'`: LQG Gramian,
            - `'pr'`: positive real Gramian,
            - `('br', gamma)`: bounded real Gramian,
        mu
            |Parameter values|.

        Returns
        -------
        sv
            One-dimensional |NumPy array| of singular values.
        Uh
            |NumPy array| of left singular vectors as rows.
        Vh
            |NumPy array| of right singular vectors as rows.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        assert self.parameters.assert_compatible(mu)
        if typ == 'lyap':
            cf = self.gramian('c_lrcf', mu=mu)
            of = self.gramian('o_lrcf', mu=mu)
        elif typ == 'bs':
            cf = self.gramian('bs_c_lrcf', mu=mu)
            of = self.gramian('bs_o_lrcf', mu=mu)
        elif typ == 'lqg':
            cf = self.gramian('lqg_c_lrcf', mu=mu)
            of = self.gramian('lqg_o_lrcf', mu=mu)
        elif typ == 'pr':
            cf = self.gramian('pr_c_lrcf', mu=mu)
            of = self.gramian('pr_o_lrcf', mu=mu)
        elif isinstance(typ, tuple) and typ[0] == 'br' and typ[1] > 0:
            gamma = typ[1]
            cf = self.gramian(('br_c_lrcf', gamma), mu=mu)
            of = self.gramian(('br_o_lrcf', gamma), mu=mu)
        else:
            raise ValueError(f'Unknown typ ({typ}).')
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
        hsv = self.presets['hsv'] if 'hsv' in self.presets else self._sv_U_V(mu=mu)[0]
        assert isinstance(hsv, np.ndarray)
        assert hsv.ndim == 1

        return hsv

    @cached
    def _h2_norm(self, mu=None):
        D_norm2 = np.sum(self.D.as_range_array(mu=mu).norm2())
        if D_norm2 != 0 and self.sampling_time == 0:
            self.logger.warning('The D operator is not exactly zero '
                                f'(squared Frobenius norm is {D_norm2}).')
            D_norm2 = 0
        assert self.parameters.assert_compatible(mu)
        if self.dim_input <= self.dim_output:
            cf = self.gramian('c_lrcf', mu=mu)
            return np.sqrt(self.C.apply(cf, mu=mu).norm2().sum() + D_norm2)
        else:
            of = self.gramian('o_lrcf', mu=mu)
            return np.sqrt(self.B.apply_adjoint(of, mu=mu).norm2().sum() + D_norm2)

    def h2_norm(self, mu=None):
        r"""Compute the :math:`\mathcal{H}_2`-norm of the |LTIModel|.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        norm
            :math:`\mathcal{H}_2`-norm.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        h2_norm = self.presets['h2_norm'] if 'h2_norm' in self.presets else self._h2_norm(mu=mu)
        assert h2_norm >= 0

        return h2_norm

    @defaults('tol')
    def hinf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False, tol=1e-10):
        r"""Compute the :math:`\mathcal{H}_\infty`-norm of the |LTIModel|.

        .. note::
            Assumes the system is asymptotically stable. Under this is assumption the
            :math:`\mathcal{H}_\infty`-norm is equal to the :math:`\mathcal{L}_\infty`-norm.
            Accordingly, this method calls :meth:`~pymor.models.iosys.LTIModel.linf_norm`.

        Parameters
        ----------
        mu
            |Parameter values|.
        return_fpeak
            Whether to return the frequency at which the maximum is achieved.
        ab13dd_equilibrate
            Whether `slycot.ab13dd` should use equilibration.
        tol
            Tolerance in norm computation.

        Returns
        -------
        norm
            :math:`\mathcal{H}_\infty`-norm.
        fpeak
            Frequency at which the maximum is achieved (if `return_fpeak` is `True`).
        """
        if 'hinf_norm' in self.presets:
            hinf_norm = self.presets['hinf_norm']
        elif not return_fpeak:
            hinf_norm = self.linf_norm(mu=mu, ab13dd_equilibrate=ab13dd_equilibrate, tol=tol)
        else:
            hinf_norm, fpeak = self.linf_norm(
                mu=mu, return_fpeak=return_fpeak, ab13dd_equilibrate=ab13dd_equilibrate, tol=tol
            )

        if return_fpeak:
            assert isinstance(fpeak, Number)
            assert hinf_norm >= 0
            return hinf_norm, fpeak
        else:
            assert hinf_norm >= 0
            return hinf_norm

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

    @cached
    def _l2_norm(self, mu=None):
        assert self.parameters.assert_compatible(mu)

        A, B, C, D, E = (op.assemble(mu=mu) for op in [self.A, self.B, self.C, self.D, self.E])
        options_lrcf = self.solver_options.get('lyap_lrcf') if self.solver_options else None

        ast_spectrum = self.get_ast_spectrum(mu=mu)

        if len(ast_spectrum[0]) == 0:
            return self.h2_norm()

        K = bernoulli_stabilize(A, E, C.as_source_array(mu=mu), ast_spectrum, trans=False)
        KC = LowRankOperator(K, np.eye(len(K)), C.as_source_array(mu=mu))

        if not isinstance(D, ZeroOperator):
            BmKD = B - LowRankOperator(K, np.eye(len(K)), D.as_source_array(mu=mu))
        else:
            BmKD = B

        solve_lyap_lrcf = solve_cont_lyap_lrcf if self.sampling_time == 0 else solve_disc_lyap_lrcf
        if self.dim_input <= self.dim_output:
            cf = solve_lyap_lrcf(A - KC, E, BmKD.as_range_array(mu=mu),
                                 trans=False, options=options_lrcf)
            return np.sqrt(self.C.apply(cf, mu=mu).norm2().sum())
        else:
            of = solve_lyap_lrcf(A - KC, E, C.as_source_array(mu=mu),
                                 trans=True, options=options_lrcf)
            return np.sqrt(BmKD.apply_adjoint(of, mu=mu).norm2().sum())

    def l2_norm(self, mu=None):
        r"""Compute the :math:`\mathcal{L}_2`-norm of the |LTIModel|.

        The :math:`\mathcal{L}_2`-norm of an |LTIModel| is defined via the integral

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
        mu
            |Parameter|.

        Returns
        -------
        norm
            :math:`\mathcal{L}_2`-norm.
        """
        if not isinstance(mu, Mu):
            mu = self.parameters.parse(mu)
        l2_norm = self.presets['l2_norm'] if 'l2_norm' in self.presets else self._l2_norm(mu=mu)
        assert l2_norm >= 0

        return l2_norm

    @cached
    def _linf_norm(self, mu=None, ab13dd_equilibrate=False, tol=1e-10):
        if 'fpeak' in self.presets:
            return spla.norm(self.transfer_function.eval_tf(self.presets['fpeak']), ord=2), self.presets['fpeak']
        elif not config.HAVE_SLYCOT:
            raise NotImplementedError

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
        return ab13dd(dico, jobe, equil, jobd, self.order, self.dim_input, self.dim_output, A, E, B, C, D, tol=tol)

    @defaults('tol')
    def linf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False, tol=1e-10):
        r"""Compute the :math:`\mathcal{L}_\infty`-norm of the |LTIModel|.

        The :math:`\mathcal{L}_\infty`-norm of an |LTIModel| is defined via

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
        tol
            Tolerance in norm computation.

        Returns
        -------
        norm
            :math:`\mathcal{L}_\infty`-norm.
        fpeak
            Frequency at which the maximum is achieved (if `return_fpeak` is `True`).
        """
        if not return_fpeak and 'linf_norm' in self.presets:
            linf_norm = self.presets['linf_norm']
        elif not return_fpeak:
            linf_norm = self.linf_norm(mu=mu, return_fpeak=True, ab13dd_equilibrate=ab13dd_equilibrate, tol=tol)[0]
        elif {'fpeak', 'linf_norm'} <= self.presets.keys():
            linf_norm, fpeak = self.presets['linf_norm'], self.presets['fpeak']
        else:
            linf_norm, fpeak = self._linf_norm(mu=mu, ab13dd_equilibrate=ab13dd_equilibrate, tol=tol)

        if return_fpeak:
            assert isinstance(fpeak, Number)
            assert linf_norm >= 0
            return linf_norm, fpeak
        else:
            assert linf_norm >= 0
            return linf_norm

    @cached
    def get_ast_spectrum(self, mu=None):
        """Compute anti-stable subset of the poles of the |LTIModel|.

        Parameters
        ----------
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

        if self.ast_pole_data is not None:
            if isinstance(E, IdentityOperator):
                E = None
            if isinstance(self.ast_pole_data, dict):
                ew, rev = eigs(A, E, left_evp=False, **self.ast_pole_data)
                ast_idx = (ew.real > 0)
                ast_ews = ew[ast_idx]
                if len(ast_ews) == 0:
                    return self.solution_space.empty(), np.empty((0,)), self.solution_space.empty()

                ast_levs = A.source.empty(reserve=len(ast_ews))
                for ae in ast_ews:
                    # l=3 avoids issues with complex conjugate pairs
                    _, lev = eigs(A, E, k=1, l=3, sigma=ae, left_evp=True)
                    ast_levs.append(lev)
                return ast_levs, ast_ews, rev[ast_idx]
            elif isinstance(self.ast_pole_data, list):
                assert all(np.real(self.ast_pole_data) > 0)
                ast_pole_data = np.sort(self.ast_pole_data)
                ast_levs = A.source.empty(reserve=len(ast_pole_data))
                ast_revs = A.source.empty(reserve=len(ast_pole_data))
                for ae in ast_pole_data:
                    _, lev = eigs(A, E, k=1, l=3, sigma=ae, left_evp=True)
                    ast_levs.append(lev)
                    _, rev = eigs(A, E, k=1, l=3, sigma=ae)
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

            A = to_matrix(A, format='dense')
            E = None if isinstance(E, IdentityOperator) else to_matrix(E, format='dense')
            ew, lev, rev = spla.eig(A, E, left=True)
            ast_idx = (ew.real > 0)
            ast_ews = ew[ast_idx]
            idx = ast_ews.argsort()

            ast_lev = self.A.source.from_numpy(lev[:, ast_idx][:, idx].T)
            ast_rev = self.A.range.from_numpy(rev[:, ast_idx][:, idx].T)

            return ast_lev, ast_ews[idx], ast_rev

    def moebius_substitution(self, M, sampling_time=0):
        r"""Create a transformed |LTIModel| by applying an arbitrary Moebius transformation.

        This method returns a transformed |LTIModel| such that the transfer function of the original
        and transformed |LTIModel| are related by a Moebius substitution of the frequency argument:

        .. math::
            H(s)=\tilde{H}(M(s)),

        where

        .. math::
            M(s) = \frac{as+b}{cs+d}

        is a Moebius transformation. See :cite:`CCA96` for details.

        Parameters
        ----------
        M
            The |MoebiusTransformation| that defines the frequency mapping.
        sampling_time
            The sampling time of the transformed system (in seconds). `0` if the system is
            continuous-time, otherwise a positive number. Defaults to zero.

        Returns
        -------
        sys
            The transformed |LTIModel|.
        """
        assert isinstance(M, MoebiusTransformation)

        a, b, c, d = MoebiusTransformation(M.coefficients, normalize=True).coefficients

        Et = a * self.E - c * self.A
        At = d * self.A - b * self.E
        C = VectorArrayOperator(Et.apply_inverse_adjoint(self.C.H.as_range_array())).H
        Ct = C @ self.E
        Dt = self.D + c * C @ self.B

        return LTIModel(At, self.B, Ct, D=Dt, E=Et, sampling_time=sampling_time)

    def to_discrete(self, sampling_time, method='Tustin', w0=0):
        """Converts a continuous-time |LTIModel| to a discrete-time |LTIModel|.

        Parameters
        ----------
        sampling_time
            A positive number that denotes the sampling time of the resulting system (in seconds).
        method
            A string that defines the transformation method. At the moment only Tustin's method is
            supported.
        w0
            If `method=='Tustin'`, this parameter can be used to specify the prewarping-frequency.
            Defaults to zero.

        Returns
        -------
        sys
            Discrete-time |LTIModel|.
        """
        if method != 'Tustin':
            return NotImplementedError
        assert self.sampling_time == 0
        sampling_time = float(sampling_time)
        assert sampling_time > 0
        assert isinstance(w0, Number)
        x = 2 / sampling_time if w0 == 0 else w0 / np.tan(w0 * sampling_time / 2)
        c2d = BilinearTransformation(x)
        return self.moebius_substitution(c2d, sampling_time=sampling_time)

    def to_continuous(self, method='Tustin', w0=0):
        """Converts a discrete-time |LTIModel| to a continuous-time |LTIModel|.

        Parameters
        ----------
        method
            A string that defines the transformation method. At the moment only Tustin's method is
            supported.
        w0
            If `method=='Tustin'`, this parameter can be used to specify the prewarping-frequency.
            Defaults to zero.

        Returns
        -------
        sys
            Continuous-time |LTIModel|.
        """
        if method != 'Tustin':
            return NotImplementedError
        assert self.sampling_time > 0
        assert isinstance(w0, Number)
        x = 2 / self.sampling_time if w0 == 0 else w0 / np.tan(w0 * self.sampling_time / 2)
        d2c = BilinearTransformation(x).inverse()
        return self.moebius_substitution(d2c, sampling_time=0)


class PHLTIModel(LTIModel):
    r"""Class for (continuous) port-Hamiltonian linear time-invariant systems.

    This class describes input-state-output systems given by

    .. math::
        E(\mu) \dot{x}(t, \mu) & = (J(\mu) - R(\mu)) Q(\mu)   x(t, \mu) + (G(\mu) - P(\mu)) u(t), \\
                     y(t, \mu) & = (G(\mu) + P(\mu))^T Q(\mu) x(t, \mu) + (S(\mu) - N(\mu)) u(t),

    where :math:`H(\mu) = Q(\mu)^T E(\mu)`,

    .. math::
        \Gamma(\mu) =
        \begin{bmatrix}
            J(\mu) & G(\mu) \\
            -G(\mu)^T & N(\mu)
        \end{bmatrix},
        \text{ and }
        \mathcal{W}(\mu) =
        \begin{bmatrix}
            R(\mu) & P(\mu) \\
            P(\mu)^T & S(\mu)
        \end{bmatrix}

    satisfy
    :math:`H(\mu) = H(\mu)^T \succ 0`,
    :math:`\Gamma(\mu)^T = -\Gamma(\mu)`, and
    :math:`\mathcal{W}(\mu) = \mathcal{W}(\mu)^T \succcurlyeq 0`.

    A dynamical system of this form, together with a given quadratic (energy) function
    :math:`\mathcal{H}(x, \mu) = \tfrac{1}{2} x^T H(\mu) x`, typically called Hamiltonian,
    is called a port-Hamiltonian system.

    All methods related to the transfer function
    (e.g., frequency response calculation and Bode plots)
    are attached to the `transfer_function` attribute.

    Parameters
    ----------
    J
        The |Operator| J.
    R
        The |Operator| R.
    G
        The |Operator| G.
    P
        The |Operator| P or `None` (then P is assumed to be zero).
    S
        The |Operator| S or `None` (then S is assumed to be zero).
    N
        The |Operator| N or `None` (then N is assumed to be zero).
    E
        The |Operator| E or `None` (then E is assumed to be identity).
    Q
        The |Operator| Q or `None` (then Q is assumed to be identity).
    solver_options
        The solver options to use to solve the Lyapunov equations.
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
    J
        The |Operator| J.
    R
        The |Operator| R.
    G
        The |Operator| G.
    P
        The |Operator| P.
    S
        The |Operator| S.
    N
        The |Operator| N.
    E
        The |Operator| E.
    Q
        The |Operator| Q.
    transfer_function
        The transfer function.
    """

    cache_region = 'memory'

    def __init__(self, J, R, G, P=None, S=None, N=None, E=None, Q=None,
                 solver_options=None, error_estimator=None, visualizer=None, name=None):
        assert J.linear
        assert J.source == J.range

        assert R.linear
        assert R.source == J.source
        assert R.source == R.range

        assert G.linear
        assert G.range == J.source

        P = P or ZeroOperator(G.range, G.source)
        assert P.linear
        assert P.range == J.source
        assert P.source == G.source

        S = S or ZeroOperator(G.source, G.source)
        assert S.linear
        assert S.source == G.source
        assert S.range == S.source

        N = N or ZeroOperator(G.source, G.source)
        assert N.linear
        assert N.source == G.source
        assert N.range == N.source

        E = E or IdentityOperator(J.source)
        assert E.linear
        assert E.source == E.range
        assert E.source == J.source

        Q = Q or IdentityOperator(J.source)
        assert Q.linear
        assert Q.source == Q.range
        assert Q.source == J.source

        super().__init__(A=J - R if isinstance(Q, IdentityOperator) else contract((J - R) @ Q),
                         B=G - P,
                         C=(G + P).H if isinstance(Q, IdentityOperator) else (G + P).H @ Q,
                         D=S - N, E=E,
                         solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer,
                         name=name)
        self.__auto_init(locals())

    def to_berlin_form(self):
        """Convert the |PHLTIModel| into its Berlin form.

        Returns a |PHLTIModel| with :math:`Q=I`, by left multiplication with :math:`Q^T`.

        Returns
        -------
        model
            |PHLTIModel| with :math:`Q=I`.
        """
        if isinstance(self.Q, IdentityOperator):
            return self

        E = contract(expand(self.Q.H @ self.E))
        J = contract(expand(self.Q.H @ self.J @ self.Q))
        R = contract(expand(self.Q.H @ self.R @ self.Q))
        G = contract(expand(self.Q.H @ self.G))
        P = contract(expand(self.Q.H @ self.P))

        return self.with_(E=E, J=J, R=R, G=G, P=P, Q=None)

    @classmethod
    def from_passive_LTIModel(cls, model):
        """
        Convert a passive |LTIModel| to a |PHLTIModel|.

        .. note::
            The method uses dense computations and converts `model` to dense matrices.

        Parameters
        ----------
        model
            The passive |LTIModel| to convert.
        generalized
            If `True`, the resulting |PHLTIModel| will have :math:`Q=I`.
        """
        # Determine solution of KYP inequality
        X = model.gramian('pr_o_dense')
        A, B, C, D, E = model.to_matrices()

        XinvAT = np.linalg.solve(X, A.T)
        AXinv = XinvAT.T # X is symmetric
        XinvCT = np.linalg.solve(X, C.T)

        Q = X
        J = 0.5 * (AXinv - XinvAT)
        R = -0.5 * (AXinv + XinvAT)
        G = 0.5 * (XinvCT + B)
        P = 0.5 * (XinvCT - B)
        S = 0.5 * (D + D.T)
        N = 0.5 * (D - D.T)

        return PHLTIModel.from_matrices(J, R, G, P=P, S=S, N=N, E=E, Q=Q, solver_options=model.solver_options,
                                        error_estimator=model.error_estimator, visualizer=model.visualizer,
                                        name=model.name)

    def __str__(self):
        string = (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    number of equations: {self.order}\n'
            f'    number of inputs:    {self.dim_input}\n'
            f'    number of outputs:   {self.dim_output}\n'
        )
        string += '    continuous-time\n'
        string += (
            f'    port-Hamiltonian\n'
            f'    linear time-invariant\n'
            f'    solution_space:  {self.solution_space}'
        )
        return string

    @classmethod
    def from_matrices(cls, J, R, G, P=None, S=None, N=None, E=None, Q=None,
                      state_id=None, solver_options=None, error_estimator=None,
                      visualizer=None, name=None):
        """Create |PHLTIModel| from matrices.

        Parameters
        ----------
        J
            The |NumPy array| or |SciPy spmatrix| J.
        R
            The |NumPy array| or |SciPy spmatrix| R.
        G
            The |NumPy array| or |SciPy spmatrix| G.
        P
            The |NumPy array| or |SciPy spmatrix| P or `None` (then P is assumed to be zero).
        S
            The |NumPy array| or |SciPy spmatrix| S or `None` (then S is assumed to be zero).
        N
            The |NumPy array| or |SciPy spmatrix| N or `None` (then N is assumed to be zero).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (then E is assumed to be identity).
        Q
            The |NumPy array| or |SciPy spmatrix| Q or `None` (then Q is assumed to be identity).
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
        phlti
            The |PHLTIModel| with operators J, R, G, P, S, N, and E.
        """
        assert isinstance(J, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(R, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(G, (np.ndarray, sps.spmatrix, sps.sparray))
        assert P is None or isinstance(P, (np.ndarray, sps.spmatrix, sps.sparray))
        assert S is None or isinstance(S, (np.ndarray, sps.spmatrix, sps.sparray))
        assert N is None or isinstance(N, (np.ndarray, sps.spmatrix, sps.sparray))
        assert E is None or isinstance(E, (np.ndarray, sps.spmatrix, sps.sparray))
        assert Q is None or isinstance(Q, (np.ndarray, sps.spmatrix, sps.sparray))

        J = NumpyMatrixOperator(J, source_id=state_id, range_id=state_id)
        R = NumpyMatrixOperator(R, source_id=state_id, range_id=state_id)
        G = NumpyMatrixOperator(G, range_id=state_id)
        if P is not None:
            P = NumpyMatrixOperator(P, range_id=state_id)
        if S is not None:
            S = NumpyMatrixOperator(S)
        if N is not None:
            N = NumpyMatrixOperator(N)
        if E is not None:
            E = NumpyMatrixOperator(E, source_id=state_id, range_id=state_id)
        if Q is not None:
            Q = NumpyMatrixOperator(Q, source_id=state_id, range_id=state_id)

        return cls(J=J, R=R, G=G, P=P, S=S, N=N, E=E, Q=Q,
                   solver_options=solver_options, error_estimator=error_estimator, visualizer=visualizer,
                   name=name)

    def to_matrices(self, format=None, mu=None):
        """Return operators as matrices.

        Parameters
        ----------
        format
            Format of the resulting matrices: |NumPy array| if 'dense',
            otherwise the appropriate |SciPy spmatrix|.
            If `None`, a choice between dense and sparse format is
            automatically made.
        mu
            The |parameter values| for which to convert the operators.

        Returns
        -------
        J
            The |NumPy array| or |SciPy spmatrix| J.
        R
            The |NumPy array| or |SciPy spmatrix| R.
        G
            The |NumPy array| or |SciPy spmatrix| G.
        P
            The |NumPy array| or |SciPy spmatrix| P or `None` (if P is a `ZeroOperator`).
        S
            The |NumPy array| or |SciPy spmatrix| S or `None` (if S is a `ZeroOperator`).
        N
            The |NumPy array| or |SciPy spmatrix| N or `None` (if N is a `ZeroOperator`).
        E
            The |NumPy array| or |SciPy spmatrix| E or `None` (if E is an `IdentityOperator`).
        Q
            The |NumPy array| or |SciPy spmatrix| Q  or `None` (if Q is an `IdentityOperator`).
        """
        J = to_matrix(self.J, format, mu)
        R = to_matrix(self.R, format, mu)
        G = to_matrix(self.G, format, mu)
        P = None if isinstance(self.P, ZeroOperator) else to_matrix(self.P, format, mu)
        S = None if isinstance(self.S, ZeroOperator) else to_matrix(self.S, format, mu)
        N = None if isinstance(self.N, ZeroOperator) else to_matrix(self.N, format, mu)
        E = None if isinstance(self.E, IdentityOperator) else to_matrix(self.E, format, mu)
        Q = None if isinstance(self.Q, IdentityOperator) else to_matrix(self.Q, format, mu)

        return J, R, G, P, S, N, E, Q

    def __add__(self, other):
        if not isinstance(other, PHLTIModel):
            return super().__add__(other)

        assert self.S.source == other.S.source

        J = BlockDiagonalOperator([self.J, other.J])
        R = BlockDiagonalOperator([self.R, other.R])
        G = BlockColumnOperator([self.G, other.G])
        P = BlockColumnOperator([self.P, other.P])
        S = self.S + other.S
        N = self.N + other.N
        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            E = BlockDiagonalOperator([self.E, other.E])
        if isinstance(self.Q, IdentityOperator) and isinstance(other.Q, IdentityOperator):
            Q = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            Q = BlockDiagonalOperator([self.Q, other.Q])

        return self.with_(J=J, R=R, G=G, P=P, S=S, N=N, E=E, Q=Q)


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

    cache_region = 'memory'

    def __init__(self, M, E, K, B, Cp, Cv=None, D=None, sampling_time=0,
                 solver_options=None, error_estimator=None, visualizer=None, name=None):

        assert M.linear
        assert M.source == M.range
        assert E.linear
        assert E.source == E.range
        assert E.source == M.source
        assert K.linear
        assert K.source == K.range
        assert K.source == M.source
        assert B.linear
        assert B.range == M.source
        assert Cp.linear
        assert Cp.source == M.range

        Cv = Cv or ZeroOperator(Cp.range, Cp.source)
        assert Cv.linear
        assert Cv.source == M.range
        assert Cv.range == Cp.range

        D = D or ZeroOperator(Cp.range, B.source)
        assert D.linear
        assert D.source == B.source
        assert D.range == Cp.range

        sampling_time = float(sampling_time)
        assert sampling_time >= 0

        assert solver_options is None or solver_options.keys() <= {'lyap_lrcf', 'lyap_dense'}

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = M.source
        self.dim_output = Cp.range.dim

        s = ProjectionParameterFunctional('s')
        s_quad = ExpressionParameterFunctional('s[0]**2', parameters=s.parameters)

        K = s_quad * self.M + s * self.E + self.K
        B = self.B
        C = self.Cp + s * self.Cv
        D = self.D
        dK = 2 * s * self.M + self.E
        dB = ZeroOperator(self.B.range, self.B.source)
        dC = self.Cv
        dD = ZeroOperator(self.D.range, self.D.source)
        parameters = Parameters.of(self.M, self.E, self.K, self.B, self.Cp, self.Cv, self.D)

        self.transfer_function = FactorizedTransferFunction(
            self.dim_input, self.dim_output,
            K, B, C, D, dK, dB, dC, dD,
            parameters=parameters, sampling_time=sampling_time, name=self.name + '_transfer_function')

        self._lti_model = LTIModel(A=SecondOrderModelOperator(0, 1, -self.E, -self.K),
                                   B=BlockColumnOperator([ZeroOperator(self.B.range, self.B.source), self.B]),
                                   C=BlockRowOperator([self.Cp, self.Cv]),
                                   D=self.D,
                                   E=(IdentityOperator(BlockVectorSpace([self.M.source, self.M.source]))
                                      if isinstance(self.M, IdentityOperator) else
                                      BlockDiagonalOperator([IdentityOperator(self.M.source), self.M])),
                                   sampling_time=self.sampling_time,
                                   solver_options=self.solver_options,
                                   error_estimator=self.error_estimator,
                                   visualizer=self.visualizer,
                                   name=self.name + '_first_order')

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
                      state_id=None, solver_options=None, error_estimator=None,
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
        assert isinstance(M, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(E, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(K, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(B, (np.ndarray, sps.spmatrix, sps.sparray))
        assert isinstance(Cp, (np.ndarray, sps.spmatrix, sps.sparray))
        assert Cv is None or isinstance(Cv, (np.ndarray, sps.spmatrix, sps.sparray))
        assert D is None or isinstance(D, (np.ndarray, sps.spmatrix, sps.sparray))

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

    def to_matrices(self, mu=None):
        """Return operators as matrices.

        Parameters
        ----------
        mu
            The |parameter values| for which to convert the operators.

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
        M = to_matrix(self.M, mu=mu)
        E = to_matrix(self.E, mu=mu)
        K = to_matrix(self.K, mu=mu)
        B = to_matrix(self.B, mu=mu)
        Cp = to_matrix(self.Cp, mu=mu)
        Cv = None if isinstance(self.Cv, ZeroOperator) else to_matrix(self.Cv, mu=mu)
        D = None if isinstance(self.D, ZeroOperator) else to_matrix(self.D, mu=mu)
        return M, E, K, B, Cp, Cv, D

    @classmethod
    def from_files(cls, M_file, E_file, K_file, B_file, Cp_file, Cv_file=None, D_file=None, sampling_time=0,
                   state_id=None, solver_options=None, error_estimator=None, visualizer=None,
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

    def to_files(self, M_file, E_file, K_file, B_file, Cp_file, Cv_file=None, D_file=None, mu=None):
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
        mu
            The |parameter values| for which to write the operators to files.
        """
        if Cv_file is None and not isinstance(self.Cv, ZeroOperator):
            raise ValueError('Cv is not zero, Cv_file must be given')
        if D_file is None and not isinstance(self.D, ZeroOperator):
            raise ValueError('D is not zero, D_file must be given')

        from pymor.tools.io import save_matrix

        M, E, K, B, Cp, Cv, D = self.to_matrices(mu=mu)
        for mat, file in [(M, M_file), (E, E_file), (K, K_file),
                          (B, B_file), (Cp, Cp_file), (Cv, Cv_file), (D, D_file)]:
            if mat is None:
                continue
            save_matrix(file, mat)

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
        return self._lti_model

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
                For `'*_dense'` types, the method assumes that the underlying Lyapunov equation
                has a unique solution, i.e. no pair of system poles adds to zero in the
                continuous-time case and no pair of system poles multiplies to one in the
                discrete-time case.
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
            return self.to_lti().gramian(typ[1:], mu=mu).blocks[0 if typ.startswith('p') else 1].copy()
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

    def h2_norm(self, mu=None):
        r"""Compute the :math:`\mathcal{H}_2`-norm.

        .. note::
            Assumes the system is asymptotically stable.

        Parameters
        ----------
        mu
            |Parameter values|.

        Returns
        -------
        norm
            :math:`\mathcal{H}_2`-norm.
        """
        return self.to_lti().h2_norm(mu=mu)

    @defaults('tol')
    def hinf_norm(self, mu=None, return_fpeak=False, ab13dd_equilibrate=False, tol=1e-10):
        r"""Compute the :math:`\mathcal{H}_\infty`-norm.

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
        tol
            Tolerance in norm computation.

        Returns
        -------
        norm
            :math:`\mathcal{H}_\infty`.
        fpeak
            Frequency at which the maximum is achieved (if `return_fpeak` is `True`).
        """
        return self.to_lti().hinf_norm(mu=mu,
                                       return_fpeak=return_fpeak,
                                       ab13dd_equilibrate=ab13dd_equilibrate,
                                       tol=tol)

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

    cache_region = 'memory'

    def __init__(self, A, Ad, tau, B, C, D=None, E=None, sampling_time=0,
                 error_estimator=None, visualizer=None, name=None):

        assert A.linear
        assert A.source == A.range
        assert isinstance(Ad, tuple)
        assert len(Ad) > 0
        assert all(Ai.linear for Ai in Ad)
        assert all(Ai.source == Ai.range for Ai in Ad)
        assert all(Ai.source == A.source for Ai in Ad)
        assert isinstance(tau, tuple)
        assert len(tau) == len(Ad)
        assert all(taui > 0 for taui in tau)
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

        super().__init__(dim_input=B.source.dim, error_estimator=error_estimator, visualizer=visualizer, name=name)
        self.__auto_init(locals())
        self.solution_space = A.source
        self.dim_output = C.range.dim
        self.q = len(Ad)

        s = ProjectionParameterFunctional('s')
        exp_tau_s = lambda taui: ExpressionParameterFunctional(f'exp(- {taui} * s[0])', parameters=s.parameters)

        K = LincombOperator((E, A) + Ad, (s, -1) + tuple(-exp_tau_s(taui) for taui in self.tau))
        B = self.B
        C = self.C
        D = self.D
        dK = LincombOperator((E,) + Ad, (1,) + tuple(taui * exp_tau_s(taui) for taui in self.tau))
        dB = ZeroOperator(self.B.range, self.B.source)
        dC = ZeroOperator(self.C.range, self.C.source)
        dD = ZeroOperator(self.D.range, self.D.source)
        parameters = Parameters.of(self.A, self.Ad, self.B, self.C, self.D, self.E)

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
        """Add an |LTIModel|, |SecondOrderModel|, |PHLTIModel|, or |LinearDelayModel|."""
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

        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
            E = BlockDiagonalOperator([self.E, other.E])
        A = BlockDiagonalOperator([self.A, other.A])
        B = BlockColumnOperator([self.B, other.B])
        C = BlockRowOperator([self.C, other.C])
        D = self.D + other.D
        return self.with_(E=E, A=A, Ad=Ad, tau=tau, B=B, C=C, D=D)

    def __radd__(self, other):
        """Add to an |LTIModel|, a |SecondOrderModel|, or a |PHLTIModel|."""
        if isinstance(other, LTIModel):
            return self + other
        elif isinstance(other, (SecondOrderModel, PHLTIModel)):
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

        if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
            E = IdentityOperator(BlockVectorSpace([self.solution_space, other.solution_space]))
        else:
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
            if isinstance(self.E, IdentityOperator) and isinstance(other.E, IdentityOperator):
                E = IdentityOperator(BlockVectorSpace([other.solution_space, self.solution_space]))
            else:
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

    cache_region = 'memory'

    def __init__(self, A, As, B, C, D=None, E=None, sampling_time=0,
                 error_estimator=None, visualizer=None, name=None):

        assert A.linear
        assert A.source == A.range
        assert isinstance(As, tuple)
        assert len(As) > 0
        assert all(Ai.linear for Ai in As)
        assert all(Ai.source == Ai.range for Ai in As)
        assert all(Ai.source == A.source for Ai in As)
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

    cache_region = 'memory'

    def __init__(self, A, N, B, C, D, E=None, sampling_time=0,
                 error_estimator=None, visualizer=None, name=None):

        assert A.linear
        assert A.source == A.range
        assert B.linear
        assert B.range == A.source
        assert isinstance(N, tuple)
        assert len(N) == B.source.dim
        assert all(Ni.linear for Ni in N)
        assert all(Ni.source == Ni.range for Ni in N)
        assert all(Ni.source == A.source for Ni in N)
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


def _lti_to_poles_b_c(lti, mu=None):
    """Compute poles and residues.

    Parameters
    ----------
    lti
        |LTIModel| consisting of |Operators| that can be converted to |NumPy arrays|.
        The D operator is ignored.
    mu
        |Parameter values|.

    Returns
    -------
    poles
        1D |NumPy array| of poles.
    b
        |NumPy array| of shape `(lti.order, lti.dim_input)`.
    c
        |NumPy array| of shape `(lti.order, lti.dim_output)`.
    """
    A = to_matrix(lti.A, format='dense', mu=mu)
    B = to_matrix(lti.B, format='dense', mu=mu)
    C = to_matrix(lti.C, format='dense', mu=mu)
    if isinstance(lti.E, IdentityOperator):
        poles, X = spla.eig(A)
        EX = X
    else:
        E = to_matrix(lti.E, format='dense', mu=mu)
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


class StepFunction(Function):

    dim_domain = 1

    def __init__(self, dim_input, component, sampling_time):
        super().__init__()
        self.__auto_init(locals())
        self.shape_range = (dim_input,)

    def evaluate(self, x, mu=None):
        e = np.zeros(self.dim_input)
        e[self.component] = self.sampling_time if self.sampling_time > 0 else 1
        return e

    def _cache_key_reduce(self):
        return ('StepFunction', self.dim_input, self.component, self.sampling_time)


class ImpulseFunction(Function):

    dim_domain = 1

    def __init__(self, dim_input, component, sampling_time):
        super().__init__()
        self.__auto_init(locals())
        self.shape_range = (dim_input,)

    def evaluate(self, x, mu=None):
        e = np.zeros(self.dim_input)
        if x[0] == 0:
            e[self.component] = self.sampling_time
        return e

    def _cache_key_reduce(self):
        return ('ImpulseFunction', self.dim_input, self.component, self.sampling_time)
