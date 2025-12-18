# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGARegressor
from pymor.core.base import BasicObject
from pymor.models.data_driven import DataDrivenInstationaryModel, DataDrivenModel


class DataDrivenReductor(BasicObject):
    """Reductor relying on a machine learning surrogate.

    The reductor works for stationary as well as for instationary
    problems and returns a suitable model for the respective case.

    Depending on the argument `target_quantity`, this reductor
    either approximates the solution or the output as a parametric
    quantity by training a machine learning surrogate.

    In case of an approximation of the solution, the reductor either
    takes a precomputed reduced basis or constructs a reduced basis
    using proper orthogonal decomposition. It then trains a machine
    learning regressor that approximates the mapping from
    parameter space to coefficients of the full-order solution
    in the reduced basis. Moreover, the reductor also works without
    providing a full-order model, in which case it requires a set of
    training parameters and corresponding solution snapshots. This way,
    the reductor can be used in a completely data-driven manner.
    The approach is described in :cite:`HU18`.

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
        |VectorArray| to use for the training of the regressor.
        Contains the solutions or outputs associated to the parameters in
        `training_parameters`.
        In the case of a time-dependent problem, the snapshots are assumed to be
        equidistant in time.
    regressor
        Regressor with `fit` and `predict` methods similar to scikit-learn
        regressors that is trained in the `reduce`-method.
    target_quantity
        Either `'solution'` or `'output'`, determines which quantity to learn.
    reduced_basis
        |VectorArray| of basis vectors of the reduced space that is used for
        reconstruction when the solution is the target quantity. If `None`,
        the result of the regressor is returned by `reconstruct`.
    T
        In the instationary case, determines the final time until which to solve.
    time_vectorized
        In the instationary case, determines whether to predict the whole time
        trajectory at once (time-vectorized version; output of the regressor is
        typically very high-dimensional in this case) or if the result for a
        single point in time is approximated (time serves as an additional input
        to the regressor).
    input_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the parameters before passing them to the regressor.
    output_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the outputs (reduced coeffcients or output quantities)
        before passing them to the regressor.
    """

    def __init__(self, training_parameters, training_snapshots,
                 regressor=VKOGARegressor(GaussianKernel()), target_quantity='solution',
                 reduced_basis=None, T=None, time_vectorized=False,
                 input_scaler=None, output_scaler=None):
        assert target_quantity in ('solution', 'output')
        self.__auto_init(locals())

        self.training_data = None

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

        # compute training data
        # i.e. pairs of parameters (potentially including time) and reduced coefficients
        if self.training_data is None:
            with self.logger.block('Computing training data ...'):
                self.training_data = self.compute_data(training_parameters, snapshots=training_snapshots)
        assert self.training_data is not None
        if self.is_stationary or not self.time_vectorized:
            assert len(self.training_data) == len(training_parameters) * self.nt

    def reduce(self, **kwargs):
        """Reduce by training a machine learning surrogate.

        Parameters
        ----------
        kwargs
            Additional arguments that will be passed to the `fit` method
            of the regressor.

        Returns
        -------
        The data-driven reduced model.
        """
        # run the actual training of the regressor
        with self.logger.block('Training of machine learning method ...'):
            # fit input and output scaler if required
            if self.input_scaler is not None:
                X = [x[0] for x in self.training_data]
                self.input_scaler = self.input_scaler.fit(X)
                X = [self.input_scaler.transform(np.atleast_2d(x[0]))[0] for x in self.training_data]
            else:
                X = [x[0] for x in self.training_data]
            if self.output_scaler is not None:
                Y = [x[1] for x in self.training_data]
                self.output_scaler = self.output_scaler.fit(Y)
                Y = [self.output_scaler.transform(np.atleast_2d(x[1]))[0] for x in self.training_data]
            else:
                Y = [x[1] for x in self.training_data]
            # fit regressor to training data
            self.regressor = self.regressor.fit(X, Y, **kwargs)

        return self._build_rom()

    def compute_data(self, parameters, snapshots):
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
        name = 'DataDrivenModel'

        with self.logger.block('Building ROM ...'):
            dim_solution_space = None
            if self.target_quantity == 'solution':
                dim_solution_space = len(self.reduced_basis)
            if self.is_stationary:
                rom = DataDrivenModel(self.regressor, target_quantity=self.target_quantity,
                                      dim_solution_space=dim_solution_space, parameters=self.parameters,
                                      input_scaler=self.input_scaler, output_scaler=self.output_scaler,
                                      name=f'{name}_reduced')
            else:
                rom = DataDrivenInstationaryModel(self.T, self.nt, self.regressor, target_quantity=self.target_quantity,
                                                  dim_solution_space=dim_solution_space, parameters=self.parameters,
                                                  input_scaler=self.input_scaler, output_scaler=self.output_scaler,
                                                  time_vectorized=self.time_vectorized, name=f'{name}_reduced')
        return rom

    def extend_training_data(self, parameters, snapshots):
        """Add sequences of parameters and corresponding snapshots to the training data."""
        self.training_data.extend(self.compute_data(parameters, snapshots))

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        assert self.target_quantity == 'solution'
        if self.reduced_basis is not None:
            return self.reduced_basis.lincomb(u.to_numpy())
        return u
