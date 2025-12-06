# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class DataDrivenModel(Model):

    def __init__(self, estimator, dim_output, parameters={},
                 output_functional=None, products=None, error_estimator=None,
                 input_scaler=None, output_scaler=None,
                 visualizer=None, name=None):

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = NumpyVectorSpace(dim_output)
        self.output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), self.solution_space)
        assert self.output_functional.source == self.solution_space
        self.dim_output = self.output_functional.range.dim

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            # obtain (reduced) coordinates by passing the parameter values
            # to the estimator
            transformed_mu = mu.to_numpy()
            if self.input_scaler:
                transformed_mu = self.input_scaler.transform(mu)
            U = self.estimator.predict(transformed_mu)
            # convert plain numpy array to element of the actual solution space
            if self.output_scaler:
                U = self.output_scaler.inverse_transform(U)
            U = self.solution_space.make_array(U.T)
            data['solution'] = U
            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)


class DataDrivenInstationaryModel(Model):

    def __init__(self, T, nt, estimator, dim_output, parameters={},
                 output_functional=None, products=None, error_estimator=None,
                 input_scaler=None, output_scaler=None,
                 visualizer=None, name=None):

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.solution_space = NumpyVectorSpace(dim_output)
        output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), self.solution_space)
        assert output_functional.source == self.solution_space
        self.dim_output = output_functional.range.dim

    def _compute(self, quantities, data, mu):
        if 'solution' in quantities:
            # collect all inputs in a single tensor
            if self.input_scaler:
                inputs = np.array([self.input_scaler.transform(mu.at_time(t).to_numpy())
                                   for t in np.linspace(0., self.T, self.nt)])
            else:
                inputs = np.array([mu.at_time(t).to_numpy() for t in np.linspace(0., self.T, self.nt)])
            # pass batch of inputs to estimator
            U = self.estimator.predict(inputs)
            if self.output_scaler:
                U = self.output_scaler.inverse_transform(U)
            # convert result into element from solution space
            data['solution'] = self.solution_space.make_array(U.T)
            quantities.remove('solution')

        super()._compute(quantities, data, mu=mu)
