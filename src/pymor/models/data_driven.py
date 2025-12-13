# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class DataDrivenModel(Model):

    def __init__(self, estimator, target_quantity='solution', dim_solution_space=None, parameters={},
                 output_functional=None, products=None, error_estimator=None,
                 input_scaler=None, output_scaler=None,
                 visualizer=None, name=None):

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        if self.target_quantity == 'solution':
            assert self.dim_solution_space
            self.solution_space = NumpyVectorSpace(self.dim_solution_space)
            self.output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), self.solution_space)
            assert self.output_functional.source == self.solution_space
            self.dim_output = self.output_functional.range.dim

    def _perform_prediction(self, mu):
        transformed_mu = np.atleast_2d(mu.to_numpy())
        if self.input_scaler is not None:
            transformed_mu = self.input_scaler.transform(transformed_mu)
        U = self.estimator.predict(transformed_mu)
        if self.output_scaler is not None:
            U = self.output_scaler.inverse_transform(U)
        return U.T

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            assert self.target_quantity == 'solution'
            U = self.solution_space.make_array(self._perform_prediction(mu))
            data['solution'] = U
            quantities.remove('solution')

        if 'output' in quantities and self.target_quantity == 'output':
            data['output'] = self._perform_prediction(mu)
            quantities.remove('output')

        super()._compute(quantities, data, mu=mu)


class DataDrivenInstationaryModel(Model):

    def __init__(self, T, nt, estimator, target_quantity='solution', dim_solution_space=None, time_vectorized=False,
                 parameters={}, output_functional=None, products=None, error_estimator=None,
                 input_scaler=None, output_scaler=None, visualizer=None, name=None):

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        if self.target_quantity == 'solution':
            assert self.dim_solution_space
            self.solution_space = NumpyVectorSpace(self.dim_solution_space)
            output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), self.solution_space)
            assert output_functional.source == self.solution_space
            self.dim_output = output_functional.range.dim

    def _perform_prediction(self, mu):
        # collect all inputs in a single tensor
        if self.time_vectorized:
            inputs = mu.to_numpy()
            if self.input_scaler is not None:
                inputs = self.input_scaler.transform(mu)
        else:
            if self.input_scaler is not None:
                inputs = np.array([self.input_scaler.transform(mu.at_time(t).to_numpy())
                                   for t in np.linspace(0., self.T, self.nt)])
            else:
                inputs = np.array([mu.at_time(t).to_numpy() for t in np.linspace(0., self.T, self.nt)])
        # pass batch of inputs to estimator
        U = self.estimator.predict(inputs)
        if self.output_scaler is not None:
            U = self.output_scaler.inverse_transform(U)
        if self.time_vectorized:
            U = U.reshape((self.nt, -1))
        return U.T

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            assert self.target_quantity == 'solution'
            data['solution'] = self.solution_space.make_array(self._perform_prediction(mu))
            quantities.remove('solution')

        if 'output' in quantities and self.target_quantity == 'output':
            data['output'] = self._perform_prediction(mu)
            quantities.remove('output')

        super()._compute(quantities, data, mu=mu)
