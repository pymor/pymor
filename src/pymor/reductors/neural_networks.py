# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_TORCH:
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from pymor.algorithms.pod import pod
    from pymor.core.base import BasicObject


    class NeuralNetworkReductor(BasicObject):

        def __init__(
            self,
            fom,
            training_set,
            validation_set,
            basis_size=None,
            basis_rtol=None,
            basis_atol=None,
            pod_params=None,
            hidden_layers='[(N+P)*2, (N+P)*2]',
            activation_function=torch.tanh,
            optimizer=optim.LBFGS,
            epochs=1000,
            learning_rate=1.
        ):
            self.__auto_init(locals())

        def reduce(self):
            rb = self.build_basis(self.training_set)
            if not self.logging_disabled:
                self.logger.info('Training the neural network')

        def build_basis(self, mus):
            if not self.logging_disabled:
                self.logger.info('Building reduced basis')

            U = self.fom.solution_space.empty()
            for mu in mus:
                U.append(self.fom.solve(mu))

            reduced_basis, _ = pod(U, modes=self.basis_size, rtol=self.basis_rtol, atol=self.basis_atol)

            return reduced_basis
