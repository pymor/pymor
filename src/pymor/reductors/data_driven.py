# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.pod import pod
from pymor.algorithms.projection import project
from pymor.core.base import BasicObject
from pymor.models.data_driven import DataDrivenInstationaryModel, DataDrivenModel
from pymor.tools.random import get_rng


class DataDrivenReductor(BasicObject):

    def __init__(self, estimator, fom=None, reduced_basis=None, training_parameters=None, validation_parameters=None,
                 training_snapshots=None, validation_snapshots=None, validation_ratio=0.1, T=None, nt=1,
                 basis_size=None, rtol=0., atol=0., l2_err=0., pod_params={}, input_scaler=None, output_scaler=None):
        assert 0 < validation_ratio < 1 or validation_parameters

        self.training_data = None
        self.validation_data = None

        if not fom:
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

    def reduce(self, estimator_settings={}):
        # compute training snapshots
        if self.training_snapshots is None:
            self.compute_training_snapshots()

        # build a reduced basis using POD if necessary
        if self.reduced_basis is None:
            self.compute_reduced_basis()

        # compute training data
        # i.e. pairs of parameters (potentially including time) and reduced coefficients
        if self.training_data is None:
            self.compute_training_data()
        assert self.training_data is not None
        assert len(self.training_data) == len(self.training_parameters) * self.nt

        if self.validation_parameters is None:
            number_validation_snapshots = int(len(self.training_data) * self.validation_ratio)
            if self.is_stationary:
                # randomly shuffle training data before splitting into two sets
                get_rng().shuffle(self.training_data)
                # split training snapshots into validation and training snapshots
                self.validation_data = self.training_data[0:number_validation_snapshots]
                self.validation_parameters = [data[0] for data in self.validation_data]
                self.training_data = self.training_data[number_validation_snapshots:]
            else:
                # create blocks of timesteps for each paraneter
                blocksize = self.nt
                blocks = [self.training_data[i:i + blocksize] for i in range(0, len(self.training_data), blocksize)]
                # shuffle the blocks
                get_rng().shuffle(blocks)
                # concatenate the shuffled blocks into a single list
                self.training_data = [timesteps for parameter in blocks for timesteps in parameter]
                # split training snapshots into validation and training snapshots
                self.validation_data = self.training_data[0:number_validation_snapshots]
                self.validation_parameters = [data[0] for data in self.validation_data[::blocksize]]
                self.training_data = self.training_data[number_validation_snapshots:]
        elif self.validation_data is None:  # compute validation snapshots if not given as input
            if self.fom is None:
                assert self.validation_snapshots is not None
            else:
                if self.validation_snapshots is None:
                    self.compute_validation_snapshots()

            # compute validation data
            self.compute_validation_data()

        assert self.validation_data is not None
        assert len(self.validation_data) == len(self.validation_parameters) * self.nt

        # run the actual training of the estimator
        with self.logger.block('Training of machine learning method ...'):
            # fit input and output scaler if required
            if self.input_scaler:
                X = [x[0] for x in self.training_data]
                self.input_scaler.fit(X)
                X = [self.input_scaler.transform(x[0]) for x in self.training_data]
            else:
                X = [x[0] for x in self.training_data]
            if self.output_scaler:
                Y = [x[1] for x in self.training_data]
                self.output_scaler.fit(Y)
                Y = [self.output_scaler(x[1]) for x in self.training_data]
            else:
                Y = [x[1] for x in self.training_data]
            # fit estimator to training data
            self.estimator.fit(X, Y, **estimator_settings)

        return self._build_rom()

    def compute_training_snapshots(self):
        """Compute training snapshots for the estimator."""
        with self.logger.block('Computing training snapshots ...'):
            self.training_snapshots = self.fom.solution_space.empty()
            for mu in self.training_parameters:
                u = self.fom.solve(mu)
                self.training_snapshots.append(u)

    def compute_reduced_basis(self):
        """Compute a reduced basis using proper orthogonal decomposition."""
        # compute reduced basis via POD
        with self.logger.block('Building reduced basis ...'):
            self.reduced_basis, svals = pod(self.training_snapshots, modes=self.basis_size, rtol=self.rtol / 2.,
                                            atol=self.atol / 2., l2_err=self.l2_err / 2.,
                                            **(self.pod_params or {}))

            # compute mean square loss
            self.mse_basis = (sum(self.training_snapshots.norm2()) - sum(svals ** 2)) / len(self.training_snapshots)

    def compute_training_data(self):
        """Compute training data for the estimator using the reduced basis."""
        with self.logger.block('Computing training samples ...'):
            self.training_data = []
            for i, mu in enumerate(self.training_parameters):
                samples = self._compute_sample(mu, self.training_snapshots[i*self.nt:(i+1)*self.nt])
                self.training_data.extend(samples)

        assert self.training_data[0][1].shape[0] == len(self.reduced_basis)

    def compute_validation_snapshots(self):
        """Compute validation data for the estimator."""
        with self.logger.block('Computing validation snapshots ...'):
            self.validation_snapshots = self.fom.solution_space.empty()
            for mu in self.validation_parameters:
                u = self.fom.solve(mu)
                self.validation_snapshots.append(u)

    def compute_validation_data(self):
        """Compute validation data for the estimator using the reduced basis."""
        assert self.validation_parameters is not None
        with self.logger.block('Computing validation samples ...'):
            self.validation_data = []
            for i, mu in enumerate(self.validation_parameters):
                samples = self._compute_sample(mu, self.validation_snapshots[i*self.nt:(i+1)*self.nt])
                self.validation_data.extend(samples)

        assert self.validation_data[0][1].shape[0] == len(self.reduced_basis)

    def _compute_sample(self, mu, u=None):
        """Transform parameter and corresponding solution to |NumPy arrays|."""
        # determine the coefficients of the full-order solutions in the reduced basis to obtain
        # the training data
        if u is None:
            assert self.fom is not None
            u = self.fom.solve(mu)

        product = self.pod_params.get('product')

        # conditional expression to check for instationary solution to return self.nt solutions
        parameters = [mu] if self.is_stationary else [mu.at_time(t) for t in np.linspace(0, self.T, self.nt)]
        samples = [(mu.to_numpy(), self.reduced_basis.inner(u_t, product=product)[:, 0]) for mu, u_t in
                   zip(parameters, u, strict=True)]

        return samples

    def _build_rom(self):
        """Construct the reduced order model."""
        if self.fom:
            projected_output_functional = project(self.fom.output_functional, None, self.reduced_basis)
            parameters = self.fom.parameters
            name = self.fom.name
        else:
            projected_output_functional = None
            parameters = self.training_parameters[0].parameters()
            name = 'data_driven'

        with self.logger.block('Building ROM ...'):
            if self.is_stationary:
                rom = DataDrivenModel(self.estimator, len(self.reduced_basis), parameters=parameters,
                                      output_functional=projected_output_functional, input_scaler=self.input_scaler,
                                      output_scaler=self.output_scaler, name=f'{name}_reduced')
            else:
                rom = DataDrivenInstationaryModel(self.T, self.nt, self.estimator, len(self.reduced_basis),
                                                  parameters=parameters,
                                                  output_functional=projected_output_functional,
                                                  input_scaler=self.input_scaler, output_scaler=self.output_scaler,
                                                  name=f'{name}_reduced')
        return rom

    def reconstruct(self, u):
        """Reconstruct high-dimensional vector from reduced vector `u`."""
        assert hasattr(self, 'reduced_basis')
        return self.reduced_basis.lincomb(u.to_numpy())
