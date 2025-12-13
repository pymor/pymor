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

        if not self.logging_disabled:
            self.logger.info(f'Architecture of the neural network:\n{self}')

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


class LongShortTermMemoryNN(nn.Module, BasicObject):
    """Class for Long Short-Term Memory neural networks (LSTMs).

    This class implements neural networks for time series of input data of arbitrary length.
    The same LSTMCell is applied in each timestep and the hidden state of the former LSTMCell
    is used as input hidden state for the next cell.

    Parameters
    ----------
    input_dimension
        Dimension of the input (at a fixed time instance) of the LSTM.
    hidden_dimension
        Dimension of the hidden state of the LSTM.
    output_dimension
        Dimension of the output of the LSTM (must be smaller than `hidden_dimension`).
    number_layers
        Number of layers in the LSTM (if greater than 1, a stacked LSTM is used).
    """

    def __init__(self, input_dimension=None, hidden_dimension=10, output_dimension=1, number_layers=1):
        assert input_dimension is None or input_dimension > 0
        assert hidden_dimension > 0
        assert output_dimension > 0
        assert hidden_dimension > output_dimension
        assert number_layers > 0

        super().__init__()
        self.__auto_init(locals())

        if input_dimension is not None:
            self.lstm = nn.LSTM(input_dimension, hidden_dimension, num_layers=number_layers,
                                proj_size=output_dimension, batch_first=True).double()

        self.logger.info(f'Architecture of the neural network:\n{self}')

    def set_input_output_dimensions(self, input_dimension, output_dimension):
        self.lstm = nn.LSTM(input_dimension, self.hidden_dimension, num_layers=self.number_layers,
                            proj_size=output_dimension, batch_first=True).double()

    def forward(self, x):
        """Performs the forward pass through the neural network.

        Initializes the hidden and cell states and applies the weights of the LSTM layers
        followed by the output layer that maps from the hidden state to the output state.

        Parameters
        ----------
        x
            Input for the neural network.

        Returns
        -------
        The output of the neural network for the input x.
        """
        # perform forward pass through LSTM and return the result
        output, _ = self.lstm(x)
        return output
