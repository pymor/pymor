# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.models.interface import Model
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class DataDrivenModel(Model):
    """Class for models of stationary problems that use regressors for prediction.

    This class implements a |Model| that uses an regressor for solution
    or output approximation.

    Parameters
    ----------
    regressor
        Regressor with `fit` and `predict` methods similar to scikit-learn
        regressors that is trained in the `reduce`-method.
    target_quantity
        Either `'solution'` or `'output'`, determines which quantity to learn.
    parameters
        |Parameters| of the reduced order model (the same as used in the full-order
        model).
    dim_solution_space
        Dimension of the solution space in case that `target_quantity='solution'`.
    input_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the parameters before passing them to the regressor.
    output_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the outputs (reduced coeffcients or output quantities)
        before passing them to the regressor.
    output_functional
        |Operator| mapping a given solution to the model output. In many applications,
        this will be a |Functional|, i.e. an |Operator| mapping to scalars.
        This is not required, however.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, regressor, target_quantity='solution', parameters={}, dim_solution_space=None,
                 input_scaler=None, output_scaler=None, output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name='DataDrivenModel'):

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        assert target_quantity == 'solution' or output_functional is None

        self.__auto_init(locals())
        if self.target_quantity == 'solution':
            assert self.dim_solution_space
            self.solution_space = NumpyVectorSpace(self.dim_solution_space)
            self.output_functional = output_functional or ZeroOperator(NumpyVectorSpace(0), self.solution_space)
            assert self.output_functional.source == self.solution_space
            self.dim_output = self.output_functional.range.dim

    def _perform_prediction(self, mu):
        """Performs the prediction with correct scaling."""
        transformed_mu = np.atleast_2d(mu.to_numpy())
        if self.input_scaler is not None:
            transformed_mu = self.input_scaler.transform(transformed_mu)
        U = self.regressor.predict(transformed_mu)
        if self.output_scaler is not None:
            U = self.output_scaler.inverse_transform(U)
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


class DataDrivenInstationaryModel(DataDrivenModel):
    """Class for models of stationary problems that use regressors for prediction.

    This class implements a |Model| that uses an regressor for solution
    or output approximation.

    Parameters
    ----------
    T
        In the instationary case, determines the final time until which to solve.
    nt
        Number of time steps.
    regressor
        Regressor with `fit` and `predict` methods similar to scikit-learn
        regressors that is trained in the `reduce`-method.
    target_quantity
        Either `'solution'` or `'output'`, determines which quantity to learn.
    parameters
        |Parameters| of the reduced order model (the same as used in the full-order
        model).
    dim_solution_space
        Dimension of the solution space in case that `target_quantity='solution'`.
    input_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the parameters before passing them to the regressor.
    output_scaler
        If not `None`, a scaler object with `fit`, `transform` and
        `inverse_transform` methods similar to the scikit-learn interface can be
        used to scale the outputs (reduced coeffcients or output quantities)
        before passing them to the regressor.
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
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    def __init__(self, T, nt, regressor, target_quantity='solution', parameters={},
                 dim_solution_space=None, input_scaler=None, output_scaler=None,
                 time_vectorized=False, output_functional=None, products=None,
                 error_estimator=None, visualizer=None, name='DataDrivenModel'):
        super().__init__(regressor, target_quantity=target_quantity, parameters=parameters,
                         dim_solution_space=dim_solution_space, input_scaler=input_scaler, output_scaler=output_scaler,
                         output_functional=output_functional, products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())

    def _perform_prediction(self, mu):
        """Performs the prediction with correct scaling."""
        # collect all inputs in a single tensor
        if self.time_vectorized:
            inputs = np.atleast_2d(mu.to_numpy())
            if self.input_scaler is not None:
                inputs = self.input_scaler.transform(inputs)
        else:
            if self.input_scaler is not None:
                inputs = np.array([self.input_scaler.transform(np.atleast_2d(mu.at_time(t).to_numpy())).flatten()
                                   for t in np.linspace(0., self.T, self.nt)])
            else:
                inputs = np.array([mu.at_time(t).to_numpy() for t in np.linspace(0., self.T, self.nt)])
        # pass batch of inputs to regressor
        U = self.regressor.predict(inputs)
        if self.output_scaler is not None:
            U = self.output_scaler.inverse_transform(U)
        if self.time_vectorized:
            U = U.reshape((self.nt, -1))
        return U.T
