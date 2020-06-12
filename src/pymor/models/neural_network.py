# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
    import numpy as np

    import torch

    from pymor.models.interface import Model


    class NeuralNetworkModel(Model):

        def __init__(self, neural_network, reduced_basis, output_functional=None,
                     products=None, estimator=None, visualizer=None, name=None):

            super().__init__(products=products, estimator=estimator, visualizer=visualizer, name=name)

            self.__auto_init(locals())
            self.linear = output_functional is None or output_functional.linear
            if output_functional is not None:
                self.output_space = output_functional.range

        def _solve(self, mu=None, return_output=False):
            if not self.logging_disabled:
                self.logger.info(f'Solving {self.name} for {mu} ...')

            converted_input = torch.from_numpy(np.fromiter(mu.values(), dtype=float)).double()
            u = self.neural_network(converted_input).data.numpy()

            if return_output:
                if self.output_functional is None:
                    raise ValueError('Model has no output')
                return u, self.output_functional.apply(u, mu=mu)
            else:
                return u
