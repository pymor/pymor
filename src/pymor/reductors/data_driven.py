# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.ml.vkoga import GaussianKernel, VKOGAEstimator
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

    In case of an approximation of the solution, the reductor either
    takes a precomputed reduced basis or constructs a reduced basis
    using proper orthogonal decomposition. It then trains a machine
    learning estimator that approximates the mapping from
    parameter space to coefficients of the full-order solution
    in the reduced basis. Moreover, the reductor also works without
    providing a full-order model, in which case it requires a set of
    training parameters and corresponding solution snapshots. This way,
    the reductor can be used in a completely data-driven manner.
    The approach is described in :cite:`HU18`.

    For `target_quantity='output'`, the machine learning estimator
    directly approximates the output depending on the parameter.
    The outputs for the training parameters are either computed using
    the full-order model or can be provided as the `training_snapshots`
    argument.

    Parameters
    ----------
    estimator
        Estimator with `fit` and `predict` methods similar to scikit-learn
        estimators that is trained in the `reduce`-method.
    target_quantity
        Either `'solution'` or `'output'`, determines which quantity to learn.
    fom
        The full-order |Model| to reduce. If `None`, the `training_parameters` with
        |parameter values| and the `training_snapshots` with corresponding solution
        |VectorArrays| or outputs have to be set.
    reduced_basis
        |VectorArray| of basis vectors of the reduced space onto which to project.
        If `None`, the reduced basis is computed using the
        :meth:`~pymor.algorithms.pod.pod` method.
    training_parameters
        |Parameter values| to use for POD (in case no `reduced_basis` is provided)
        and training of the neural network.
    training_snapshots
        |VectorArray| to use for POD and training of the neural network.
        Contains the solutions to the parameters of the
        `training_parameters` and can be `None` when `fom` is not `None`.
        In the case of a time-dependent problem, the snapshots are assumed to be
        equidistant in time.
    T
        In the instationary case, determines the final time until which to solve.
    time_vectorized
        In the instationary case, determines whether to predict the whole time
        trajectory at once (time-vectorized version; output of the estimator is
        typically very high-dimensional in this case) or if the result for a
        single point in time is approximated (time serves as an additional input
        to the estimator).
    basis_size
        Desired size of the reduced basis. If `None`, rtol, atol or l2_err must
        be provided.
    rtol
        Relative tolerance the basis should guarantee on the training parameters.
    atol
        Absolute tolerance the basis should guarantee on the training parameters.
    l2_err
        L2-approximation error the basis should not exceed on the training
        parameters.
    pod_params
        Dict of additional parameters for the POD-method.
    input_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the parameters before passing them to the estimator.
    output_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the outputs (reduced coeffcients or output quantities)
        before passing them to the estimator.
    """

    def __init__(self, estimator=VKOGAEstimator(GaussianKernel()), target_quantity='solution', fom=None,
                 reduced_basis=None, training_parameters=None, training_snapshots=None, T=None,
                 time_vectorized=False, basis_size=None, rtol=0., atol=0., l2_err=0., pod_params={},
                 input_scaler=None, output_scaler=None):
        assert target_quantity in ('solution', 'output')

        self.training_data = None

        if fom is None:
            assert training_parameters is not None
            assert len(training_parameters) > 0
            assert training_snapshots is not None
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
        else:
            self.parameters_dim = fom.parameters.dim
            if hasattr(fom, 'time_stepper'):  # instationary
                self.nt = fom.time_stepper.nt + 1  # + 1 because of initial condition
                self.T = fom.T
                self.is_stationary = False
            else:  # stationary
                self.nt = 1
                self.is_stationary = True

        self.__auto_init(locals())

    def reduce(self, **kwargs):
        """Reduce by training a machine learning surrogate.

        Parameters
        ----------
        kwargs
            Additional arguments that will be passed to the `fit` method
            of the estimator.

        Returns
        -------
        The data-driven reduced model.
        """
        if self.target_quantity == 'solution' and self.training_snapshots is None:
            self.training_snapshots = self.compute_snapshots(self.training_parameters)

        # build a reduced basis using POD if necessary
        if self.target_quantity == 'solution' and self.reduced_basis is None:
            self.compute_reduced_basis()

        # compute training data
        # i.e. pairs of parameters (potentially including time) and reduced coefficients
        if self.training_data is None:
            with self.logger.block('Computing training data ...'):
                self.training_data = self.compute_data(self.training_parameters, snapshots=self.training_snapshots)
        assert self.training_data is not None
        if self.is_stationary or not self.time_vectorized:
            assert len(self.training_data) == len(self.training_parameters) * self.nt

        # run the actual training of the estimator
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
            # fit estimator to training data
            self.estimator = self.estimator.fit(X, Y, **kwargs)

        return self._build_rom()

    def compute_snapshots(self, parameters):
        """Compute snapshots for the given parameters."""
        assert self.target_quantity == 'solution'
        assert self.fom is not None
        U = self.fom.solution_space.empty(reserve=len(parameters))
        for mu in parameters:
            U.append(self.fom.solve(mu))
        return U

    def compute_data(self, parameters, snapshots=None):
        """Collect data for the estimator using the reduced basis."""
        data = []
        if snapshots is not None:
            if self.target_quantity == 'solution':
                product = self.pod_params.get('product')
                func = lambda i, mu: self.reduced_basis.inner(snapshots[i*self.nt:(i+1)*self.nt], product=product).T
            elif self.target_quantity == 'output':
                func = lambda i, mu: snapshots[i*self.nt:(i+1)*self.nt]
        else:
            assert self.fom is not None
            if self.target_quantity == 'solution':
                product = self.pod_params.get('product')
                func = lambda i, mu: self.reduced_basis.inner(self.fom.solve(mu), product=product).T
            elif self.target_quantity == 'output':
                func = lambda i, mu: self.fom.output(mu)

        def func_wrapped(i, mu):
            if not self.is_stationary or not self.time_vectorized:
                return func(i, mu)
            else:
                return func(i, mu).flatten()

        for i, mu in enumerate(parameters):
            samples = self._compute_sample(mu, func_wrapped(i, mu))
            data.extend(samples)

        return data

    def compute_reduced_basis(self):
        """Compute a reduced basis using proper orthogonal decomposition."""
        # compute reduced basis via POD
        with self.logger.block('Building reduced basis ...'):
            self.reduced_basis, svals = pod(self.training_snapshots, modes=self.basis_size, rtol=self.rtol / 2.,
                                            atol=self.atol / 2., l2_err=self.l2_err / 2.,
                                            **(self.pod_params or {}))

            # compute mean square loss
            self.mse_basis = (sum(self.training_snapshots.norm2()) - sum(svals ** 2)) / len(self.training_snapshots)

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
        projected_output_functional = None
        if self.fom is not None:
            if self.target_quantity == 'solution':
                projected_output_functional = project(self.fom.output_functional, None, self.reduced_basis)
            parameters = self.fom.parameters
            name = self.fom.name
        else:
            parameters = self.training_parameters[0].parameters()
            name = 'DataDrivenModel'

        with self.logger.block('Building ROM ...'):
            dim_solution_space = None
            if self.target_quantity == 'solution':
                dim_solution_space = len(self.reduced_basis)
            if self.is_stationary:
                rom = DataDrivenModel(self.estimator, target_quantity=self.target_quantity,
                                      dim_solution_space=dim_solution_space, parameters=parameters,
                                      output_functional=projected_output_functional, input_scaler=self.input_scaler,
                                      output_scaler=self.output_scaler, name=f'{name}_reduced')
            else:
                rom = DataDrivenInstationaryModel(self.T, self.nt, self.estimator, target_quantity=self.target_quantity,
                                                  dim_solution_space=dim_solution_space, parameters=parameters,
                                                  output_functional=projected_output_functional,
                                                  input_scaler=self.input_scaler, output_scaler=self.output_scaler,
                                                  time_vectorized=self.time_vectorized, name=f'{name}_reduced')
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        assert self.target_quantity == 'solution'
        assert hasattr(self, 'reduced_basis')
        return self.reduced_basis.lincomb(u.to_numpy())
