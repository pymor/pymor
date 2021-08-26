# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright 2013-2021 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

"""Remark on the documentation:

Due to an issue in autoapi, the classes `NeuralNetworkStatefreeOutputModel`,
`NeuralNetworkInstationaryModel`, `NeuralNetworkInstationaryStatefreeOutputModel`
and `FullyConnectedNN` do not appear in the documentation,
see https://github.com/pymor/pymor/issues/1343.
"""

from pymor.core.config import config


if config.HAVE_TORCH:
    import numpy as np

    import torch
    import torch.nn as nn

    from pymor.core.base import BasicObject
    from pymor.models.interface import Model
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class NeuralNetworkModel(Model):
        """Class for models of stationary problems that use artificial neural networks.

        This class implements a |Model| that uses a neural network for solving.

        Parameters
        ----------
        neural_network
            The neural network that approximates the mapping from parameter space
            to solution space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the reduced space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
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

        def __init__(self, neural_network, parameters={}, output_functional=None,
                     products=None, error_estimator=None, visualizer=None, name=None):

            super().__init__(products=products, error_estimator=error_estimator,
                             visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.solution_space = NumpyVectorSpace(neural_network.output_dimension)
            if output_functional is not None:
                self.dim_output = output_functional.range.dim

        def _compute_solution(self, mu=None, **kwargs):

            # convert the parameter `mu` into a form that is usable in PyTorch
            converted_input = torch.DoubleTensor(mu.to_numpy())
            # obtain (reduced) coordinates by forward pass of the parameter values
            # through the neural network
            U = self.neural_network(converted_input).data.numpy()
            # convert plain numpy array to element of the actual solution space
            U = self.solution_space.make_array(U)

            return U

    class NeuralNetworkStatefreeOutputModel(Model):
        """Class for models of the output of stationary problems that use ANNs.

        This class implements a |Model| that uses a neural network for solving for the output
        quantity.

        Parameters
        ----------
        neural_network
            The neural network that approximates the mapping from parameter space
            to output space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the output space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
        error_estimator
            An error estimator for the problem. This can be any object with
            an `estimate_error(U, mu, m)` method. If `error_estimator` is
            not `None`, an `estimate_error(U, mu)` method is added to the
            model which will call `error_estimator.estimate_error(U, mu, self)`.
        name
            Name of the model.
        """

        def __init__(self, neural_network, parameters={}, error_estimator=None, name=None):

            super().__init__(error_estimator=error_estimator, name=name)

            self.__auto_init(locals())

        def _compute(self, solution=False, output=False, solution_d_mu=False, output_d_mu=False,
                     solution_error_estimate=False, output_error_estimate=False,
                     output_d_mu_return_array=False, mu=None, **kwargs):
            if output:
                converted_input = torch.from_numpy(mu.to_numpy()).double()
                output = self.neural_network(converted_input).data.numpy()
                return {'output': output, 'solution': None}
            return {}

    class NeuralNetworkInstationaryModel(Model):
        """Class for models of instationary problems that use artificial neural networks.

        This class implements a |Model| that uses a neural network for solving.

        Parameters
        ----------
        T
            The final time T.
        nt
            The number of time steps.
        neural_network
            The neural network that approximates the mapping from parameter space
            to solution space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the reduced space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
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

        def __init__(self, T, nt, neural_network, parameters={}, output_functional=None,
                     products=None, error_estimator=None, visualizer=None, name=None):

            super().__init__(products=products, error_estimator=error_estimator,
                             visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.solution_space = NumpyVectorSpace(neural_network.output_dimension)
            if output_functional is not None:
                self.dim_output = output_functional.range.dim

        def _compute_solution(self, mu=None, **kwargs):

            U = self.solution_space.empty(reserve=self.nt)
            dt = self.T / (self.nt - 1)
            t = 0.

            # iterate over time steps
            for i in range(self.nt):
                mu = mu.with_(t=t)
                # convert the parameter `mu` into a form that is usable in PyTorch
                converted_input = torch.DoubleTensor(mu.to_numpy())
                # obtain (reduced) coordinates by forward pass of the parameter values
                # through the neural network
                result_neural_network = self.neural_network(converted_input).data.numpy()
                # convert plain numpy array to element of the actual solution space
                U.append(self.solution_space.make_array(result_neural_network))
                t += dt

            return U

    class NeuralNetworkInstationaryStatefreeOutputModel(Model):
        """Class for models of the output of instationary problems that use ANNs.

        This class implements a |Model| that uses a neural network for solving for the output
        quantity in the instationary case.

        Parameters
        ----------
        T
            The final time T.
        nt
            The number of time steps.
        neural_network
            The neural network that approximates the mapping from parameter space
            to output space. Should be an instance of
            :class:`~pymor.models.neural_network.FullyConnectedNN` with input size that
            matches the (total) number of parameters and output size equal to the
            dimension of the output space.
        parameters
            |Parameters| of the reduced order model (the same as used in the full-order
            model).
        error_estimator
            An error estimator for the problem. This can be any object with
            an `estimate_error(U, mu, m)` method. If `error_estimator` is
            not `None`, an `estimate_error(U, mu)` method is added to the
            model which will call `error_estimator.estimate_error(U, mu, self)`.
        name
            Name of the model.
        """

        def __init__(self, T, nt, neural_network, parameters={}, error_estimator=None, name=None):

            super().__init__(error_estimator=error_estimator, name=name)

            self.__auto_init(locals())

        def _compute(self, solution=False, output=False, solution_d_mu=False, output_d_mu=False,
                     solution_error_estimate=False, output_error_estimate=False,
                     output_d_mu_return_array=False, mu=None, **kwargs):

            if output:
                outputs = []
                dt = self.T / (self.nt - 1)
                t = 0.

                # iterate over time steps
                for i in range(self.nt):
                    mu = mu.with_(t=t)
                    # convert the parameter `mu` into a form that is usable in PyTorch
                    converted_input = torch.from_numpy(mu.to_numpy()).double()
                    # obtain approximate output quantity by forward pass of the parameter values
                    # through the neural network
                    result_neural_network = self.neural_network(converted_input).data.numpy()
                    # append approximate output to list of outputs
                    outputs.append(result_neural_network)
                    t += dt

                return {'output': np.array(outputs), 'solution': None}
            return {}

    class FullyConnectedNN(nn.Module, BasicObject):
        """Class for neural networks with fully connected layers.

        This class implements neural networks consisting of linear and fully connected layers.
        Furthermore, the same activation function is used between each layer, except for the
        last one where no activation function is applied.

        Parameters
        ----------
        layer_sizes
            List of sizes (i.e. number of neurons) for the layers of the neural network.
        activation_function
            Function to use as activation function between the single layers.
        """

        def __init__(self, layer_sizes, activation_function=torch.tanh):
            super().__init__()

            if layer_sizes is None or not len(layer_sizes) > 1 or not all(size >= 1 for size in layer_sizes):
                raise ValueError

            self.input_dimension = layer_sizes[0]
            self.output_dimension = layer_sizes[-1]

            self.layers = nn.ModuleList()
            self.layers.extend([nn.Linear(int(layer_sizes[i]), int(layer_sizes[i+1]))
                                for i in range(len(layer_sizes) - 1)])

            self.activation_function = activation_function

            if not self.logging_disabled:
                self.logger.info(f'Architecture of the neural network:\n{self}')

        def forward(self, x):
            """Performs the forward pass through the neural network.

            Applies the weights in the linear layers and passes the outcomes to the
            activation function.

            Parameters
            ----------
            x
                Input for the neural network.

            Returns
            -------
            The output of the neural network for the input x.
            """
            for i in range(len(self.layers) - 1):
                x = self.activation_function(self.layers[i](x))
            return self.layers[len(self.layers)-1](x)
