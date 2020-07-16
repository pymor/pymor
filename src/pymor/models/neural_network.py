# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
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
        output_functional
            |Operator| mapping a given solution to the model output. In many applications,
            this will be a |Functional|, i.e. an |Operator| mapping to scalars.
            This is not required, however.
        products
            A dict of inner product |Operators| defined on the discrete space the
            problem is posed on. For each product with key `'x'` a corresponding
            attribute `x_product`, as well as a norm method `x_norm` is added to
            the model.
        estimator
            An error estimator for the problem. This can be any object with
            an `estimate(U, mu, m)` method. If `estimator` is
            not `None`, an `estimate(U, mu)` method is added to the
            model which will call `estimator.estimate(U, mu, self)`.
        visualizer
            A visualizer for the problem. This can be any object with
            a `visualize(U, m, ...)` method. If `visualizer`
            is not `None`, a `visualize(U, *args, **kwargs)` method is added
            to the model which forwards its arguments to the
            visualizer's `visualize` method.
        name
            Name of the model.
        """

        def __init__(self, neural_network, output_functional=None, products=None,
                     estimator=None, visualizer=None, name=None):

            super().__init__(products=products, estimator=estimator, visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.solution_space = NumpyVectorSpace(neural_network.output_dimension)
            self.linear = output_functional is None or output_functional.linear
            if output_functional is not None:
                self.output_space = output_functional.range

        def _solve(self, mu=None, return_output=False):
            if not self.logging_disabled:
                self.logger.info(f'Solving {self.name} for {mu} ...')

            # convert the parameter `mu` into a form that is usable in PyTorch
            converted_input = torch.from_numpy(mu.to_numpy()).double()
            # obtain (reduced) coordinates by forward pass of the parameter values through the neural network
            U = self.neural_network(converted_input).data.numpy()
            # convert plain numpy array to element of the actual solution space
            U = self.solution_space.make_array(U)

            if return_output:
                if self.output_functional is None:
                    raise ValueError('Model has no output')
                return U, self.output_functional.apply(U, mu=mu)
            else:
                return U


    class FullyConnectedNN(nn.Module, BasicObject):
        """Class for neural networks with fully connected layers.

        This class implements neural networks consisting of linear and fully connected layers.
        Furthermore, the same activation function is used between each layer, except for the
        last one where no activation function is applied.

        Parameters
        ----------
        layers_sizes
            List of sizes (i.e. number of neurons) for the layers of the neural network.
        activation_function
            Function to use as activation function between the single layers.
        """

        def __init__(self, layers_sizes, activation_function=torch.tanh):
            super().__init__()

            if layers_sizes is None or not len(layers_sizes) > 1 or not all(size >= 1 for size in layers_sizes):
                raise ValueError

            self.input_dimension = layers_sizes[0]
            self.output_dimension = layers_sizes[-1]

            self.layers = nn.ModuleList()
            self.layers.extend([nn.Linear(int(layers_sizes[i]), int(layers_sizes[i+1]))
                                for i in range(len(layers_sizes) - 1)])

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
