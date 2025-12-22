# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

config.require('TORCH')

import torch
import torch.nn as nn

from pymor.core.base import BasicObject


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

    def __init__(self, hidden_layers, input_dimension=None, output_dimension=None, activation_function=torch.tanh):
        super().__init__()

        self.hidden_layers = hidden_layers

        if input_dimension is not None and output_dimension is not None:
            self.set_input_output_dimensions(input_dimension=input_dimension, output_dimension=output_dimension)

        self.activation_function = activation_function

    def set_input_output_dimensions(self, input_dimension, output_dimension):
        layer_sizes = [input_dimension]
        layer_sizes.extend(self.hidden_layers)
        layer_sizes.append(output_dimension)
        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(int(layer_sizes[i]), int(layer_sizes[i+1])).double()
                            for i in range(len(layer_sizes) - 1)])

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
