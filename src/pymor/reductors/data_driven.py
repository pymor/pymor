# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import inspect

import numpy as np

from pymor.algorithms.ml.vkoga import VKOGARegressor
from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project
from pymor.core.base import BasicObject
from pymor.models.data_driven import DataDrivenInstationaryModel, DataDrivenModel


class DataDrivenReductor(BasicObject):
    """Reductor relying on a machine learning surrogate.

    The reductor works for stationary as well as for instationary
    problems and returns a suitable model for the respective case.

    Depending on the argument `target_quantity`, this reductor
    either approximates the solution or the output as a parametric
    quantity by training a machine learning surrogate.

    In case of an approximation of the solution, the reductor trains
    a machine learning regressor that approximates the mapping from
    parameter space to a |NumPy array|. Typically, the array contains
    coefficients of the solution with respect to a reduced basis.

    For `target_quantity='output'`, the machine learning regressor
    directly approximates the output depending on the parameter.
    The outputs for the training parameters are either computed using
    the full-order model or can be provided as the `training_snapshots`
    argument.

    Parameters
    ----------
    training_parameters
        |Parameter values| to use for training of the regressor.
    training_snapshots
        Iterable containing the training snapshots of the regressor.
        Contains the solutions (reduced coefficients w.r.t. the reduced basis
        or outputs) associated to the parameters in `training_parameters`.
        In the case of a time-dependent problem, the snapshots are assumed to be
        equidistant in time.
    regressor
        Regressor with `fit` and `predict` methods similar to scikit-learn
        regressors that is trained in the `reduce`-method.
        Defaults to :class:`~pymor.algorithms.ml.vkoga.regressor.VKOGARegressor`.
        Alternatively, one can pass a class which will be instantiated using
        the attributes in `regressor_parameters`.
    regressor_parameters
        Dictionary with parameters for regressor instantiation. This will be used
        only when a class instead of a regressor object is passed as `regressor`.
    target_quantity
        Either `'solution'` or `'output'`, determines which quantity to learn.
    T
        In the instationary case, determines the final time until which to solve.
    time_vectorized
        In the instationary case, determines whether to predict the whole time
        trajectory at once (time-vectorized version; output of the regressor is
        typically very high-dimensional in this case) or if the result for a
        single point in time is approximated (time serves as an additional input
        to the regressor).
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    input_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the parameters before passing them to the regressor.
    output_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the outputs (reduced coeffcients or output quantities)
        before passing them to the regressor.
    input_scaler_fitted
        If `True`, the `input_scaler` is assumed to be already fitted and will
        not be refitted during :meth:`reduce`. This enables the incremental
        `extend` path for regressors that support it. Useful when the scaler
        has been pre-fitted based on domain knowledge (e.g., the parameter space
        bounds).
    output_scaler_fitted
        If `True`, the `output_scaler` is assumed to be already fitted and will
        not be refitted during :meth:`reduce`.
    """

    def __init__(self, training_parameters, training_snapshots,
                 regressor=VKOGARegressor, regressor_parameters=None, target_quantity='solution',
                 T=None, time_vectorized=False, output_functional=None,
                 input_scaler=None, output_scaler=None,
                 input_scaler_fitted=False, output_scaler_fitted=False):
        assert target_quantity in ('solution', 'output')
        assert target_quantity == 'solution' or output_functional is None
        self.__auto_init(locals())

        if inspect.isclass(self.regressor):
            if self.regressor_parameters is None:
                self.regressor_parameters = {}
            self.regressor = self.regressor(**regressor_parameters)

        assert training_parameters is not None
        assert len(training_parameters) > 0
        assert training_snapshots is not None
        self.parameters = training_parameters[0].parameters()
        self.parameters_dim = training_parameters[0].parameters().dim
        self.nt = int(len(training_snapshots) / len(training_parameters))
        assert len(training_snapshots) == len(training_parameters) * self.nt
        if self.nt > 1:  # instationary
            assert T is not None
            self.T = T
            self.is_stationary = False
        else:  # stationary
            assert T is None
            self.is_stationary = True

        self.dim_solution_space = None

        self._n_trained = 0

        # compute training data
        # i.e. pairs of parameters (potentially including time) and reduced coefficients
        with self.logger.block('Computing training data ...'):
            self.training_data = self._compute_data(training_parameters, snapshots=training_snapshots)
        if self.target_quantity == 'solution':
            self.dim_solution_space = len(self.training_data[0][1])
            if not self.is_stationary and self.time_vectorized:
                self.dim_solution_space = self.dim_solution_space // self.nt
        if self.is_stationary or not self.time_vectorized:
            assert len(self.training_data) == len(training_parameters) * self.nt

    def _scale_data(self, data):
        """Apply input and output scalers to training data."""
        X = np.array([x[0] for x in data])
        Y = np.array([x[1] for x in data])
        if self.input_scaler is not None:
            if not self.input_scaler_fitted:
                self.input_scaler = self.input_scaler.fit(X)
                self.input_scaler_fitted = True
            X = self.input_scaler.transform(X)
        if self.output_scaler is not None:
            if not self.output_scaler_fitted:
                self.output_scaler = self.output_scaler.fit(Y)
                self.output_scaler_fitted = True
            Y = self.output_scaler.transform(Y)
        return X, Y

    def reduce(self, **kwargs):
        """Reduce by training a machine learning surrogate.

        If the regressor supports incremental extension via an `extend` method
        and has already been fitted, only the new training data (added via
        :meth:`extend_training_data`) is passed to `extend`. Otherwise, the
        regressor is fully retrained on all training data.

        Incremental extension requires that all scalers are either pre-fitted
        (via the `input_scaler_fitted` / `output_scaler_fitted` flags) or not
        used, since unfitted scalers need to be refitted on the full dataset.

        Parameters
        ----------
        kwargs
            Additional arguments that will be passed to the `fit` method
            of the regressor.

        Returns
        -------
        The data-driven reduced model.
        """
        new_data = self.training_data[self._n_trained:]

        scalers_ready = ((self.input_scaler is None or self.input_scaler_fitted)
                         and (self.output_scaler is None or self.output_scaler_fitted))

        use_extend = (self._n_trained > 0
                      and len(new_data) > 0
                      and hasattr(self.regressor, 'extend')
                      and scalers_ready)

        if use_extend:
            with self.logger.block('Extending machine learning method ...'):
                X_new, Y_new = self._scale_data(new_data)
                self.regressor.extend(np.array(X_new), np.array(Y_new))
        else:
            with self.logger.block('Training of machine learning method ...'):
                X, Y = self._scale_data(self.training_data)
                self.regressor = self.regressor.fit(X, Y, **kwargs)

        self._n_trained = len(self.training_data)
        return self._build_rom()

    def _compute_data(self, parameters, snapshots):
        """Collect data for the regressor using the reduced basis."""
        data = []
        func = lambda i, mu: snapshots[i*self.nt:(i+1)*self.nt]

        def func_wrapped(i, mu):
            if not self.is_stationary or not self.time_vectorized:
                return func(i, mu)
            else:
                return func(i, mu).flatten()

        for i, mu in enumerate(parameters):
            samples = self._compute_sample(mu, func_wrapped(i, mu))
            data.extend(samples)

        return data

    def _compute_sample(self, mu, u):
        """Transform parameter and corresponding solution to |NumPy arrays|."""
        # conditional expression to check for instationary solution to return self.nt solutions
        if not self.is_stationary and not self.time_vectorized:
            parameters = [mu.at_time(t) for t in np.linspace(0, self.T, self.nt)]
            samples = [(mu.to_numpy(), u_t.flatten()) for mu, u_t in zip(parameters, u, strict=True)]
        else:
            samples = [(mu.to_numpy(), u.flatten())]

        return samples

    def _build_rom(self):
        """Construct the reduced order model."""
        with self.logger.block('Building ROM ...'):
            if self.is_stationary:
                rom = DataDrivenModel(self.regressor, target_quantity=self.target_quantity,
                                      dim_solution_space=self.dim_solution_space, parameters=self.parameters,
                                      output_functional=self.output_functional,
                                      input_scaler=self.input_scaler, output_scaler=self.output_scaler)
            else:
                rom = DataDrivenInstationaryModel(self.T, self.nt, self.regressor, target_quantity=self.target_quantity,
                                                  dim_solution_space=self.dim_solution_space,
                                                  parameters=self.parameters, output_functional=self.output_functional,
                                                  input_scaler=self.input_scaler, output_scaler=self.output_scaler,
                                                  time_vectorized=self.time_vectorized)
        return rom

    def extend_training_data(self, parameters, snapshots):
        """Add sequences of parameters and corresponding snapshots to the training data."""
        self.training_data.extend(self._compute_data(parameters, snapshots))


class DataDrivenPODReductor(DataDrivenReductor):
    """Reductor building a reduced basis and relying on a machine learning surrogate.

    In addition to the :class:`~pymor.reductors.data_driven.DataDrivenReductor`,
    this reductor uses snapshot data in order to construct a reduced basis via POD
    and projects the snapshots onto the reduced basis to generate data for the
    machine learning training. The approach is described in :cite:`HU18`.
    See :class:`~pymor.reductors.data_driven.DataDrivenReductor` for more details.

    Parameters
    ----------
    training_parameters
        |Parameter values| to use for training of the regressor.
    training_snapshots
        |VectorArray| to use for the training of the regressor.
        Contains the solutions or outputs associated to the parameters in
        `training_parameters`.
        In the case of a time-dependent problem, the snapshots are assumed to be
        equidistant in time.
    regressor
        See :class:`~pymor.reductors.data_driven.DataDrivenReductor`.
    T
        See :class:`~pymor.reductors.data_driven.DataDrivenReductor`.
    time_vectorized
        See :class:`~pymor.reductors.data_driven.DataDrivenReductor`.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
        The output functional will be projected automatically onto the reduced space.
    input_scaler
        See :class:`~pymor.reductors.data_driven.DataDrivenReductor`.
    output_scaler
        See :class:`~pymor.reductors.data_driven.DataDrivenReductor`.
    product
        Inner product |Operators| defined on the discrete space the
        problem is posed on. Used for reduced basis computation via POD and
        or orthogonal projection onto the reduced basis.
    pod_params
        Dict of additional parameters for the POD-method.
    """

    def __init__(self, training_parameters, training_snapshots, regressor=None,
                 T=None, time_vectorized=False, output_functional=None,
                 input_scaler=None, output_scaler=None, product=None,
                 pod_params=None):
        self.reduced_basis = None
        self.__auto_init(locals())

        if self.pod_params is None:
            self.pod_params = {}

    def reduce(self, **kwargs):
        if self.reduced_basis is None:
            self.reduced_basis = self._compute_reduced_basis()
            projected_training_snapshots = self.training_snapshots.inner(self.reduced_basis, product=self.product)
            projected_output_functional = None
            if self.output_functional is not None:
                projected_output_functional = project(self.output_functional, self.reduced_basis)

            super().__init__(self.training_parameters, projected_training_snapshots,
                             regressor=self.regressor, target_quantity='solution',
                             output_functional=projected_output_functional,
                             T=self.T, time_vectorized=self.time_vectorized,
                             input_scaler=self.input_scaler, output_scaler=self.output_scaler)

        return super().reduce(**kwargs)

    def _compute_reduced_basis(self):
        """Compute a reduced basis using POD."""
        return pod(self.training_snapshots, **self.pod_params)[0]

    def extend_training_data(self, parameters, snapshots):
        """Add sequences of parameters and corresponding snapshots to the training data."""
        projected_snapshots = snapshots.inner(self.reduced_basis, product=self.product)
        self.training_data.extend(self._compute_data(parameters, projected_snapshots))

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        return self.reduced_basis.lincomb(u.to_numpy())
