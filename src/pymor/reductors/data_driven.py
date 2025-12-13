# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project
from pymor.algorithms.vkoga import GaussianKernel, VKOGAEstimator
from pymor.core.base import BasicObject
from pymor.models.data_driven import DataDrivenInstationaryModel, DataDrivenModel


class DataDrivenReductor(BasicObject):

    def __init__(self, estimator=VKOGAEstimator(GaussianKernel()), target_quantity='solution', fom=None,
                 reduced_basis=None, training_parameters=None, training_snapshots=None, T=None, nt=1,
                 basis_size=None, rtol=0., atol=0., l2_err=0., pod_params={}, input_scaler=None, output_scaler=None):
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
                self.nt = fom.time_stepper.nt + 1  # +1 because of initial condition
                self.T = fom.T
                self.is_stationary = False
            else:  # stationary
                self.nt = 1
                self.is_stationary = True

        self.__auto_init(locals())

    def reduce(self, **kwargs):
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
        assert len(self.training_data) == len(self.training_parameters) * self.nt

        # run the actual training of the estimator
        with self.logger.block('Training of machine learning method ...'):
            # fit input and output scaler if required
            if self.input_scaler is not None:
                X = [x[0] for x in self.training_data]
                self.input_scaler.fit(X)
                X = [self.input_scaler.transform(np.atleast_2d(x[0]))[0] for x in self.training_data]
            else:
                X = [x[0] for x in self.training_data]
            if self.output_scaler is not None:
                Y = [x[1] for x in self.training_data]
                self.output_scaler.fit(Y)
                Y = [self.output_scaler.transform(np.atleast_2d(x[1]))[0] for x in self.training_data]
            else:
                Y = [x[1] for x in self.training_data]
            # fit estimator to training data
            self.estimator.fit(X, Y, **kwargs)

        return self._build_rom()

    def compute_snapshots(self, parameters):
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

        for i, mu in enumerate(parameters):
            samples = self._compute_sample(mu, func(i, mu))
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
        parameters = [mu] if self.is_stationary else [mu.at_time(t) for t in np.linspace(0, self.T, self.nt)]
        samples = [(mu.to_numpy(), u_t) for mu, u_t in zip(parameters, u, strict=True)]
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
            name = 'data_driven'

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
                                                  name=f'{name}_reduced')
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        assert self.target_quantity == 'solution'
        assert hasattr(self, 'reduced_basis')
        return self.reduced_basis.lincomb(u.to_numpy())
