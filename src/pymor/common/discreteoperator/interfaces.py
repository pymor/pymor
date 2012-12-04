from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import pymor.core as core

class IDiscreteOperator(core.BasicInterface):

    parameter_dim = 0

    @core.interfaces.abstractmethod
    def apply(self, U, mu=np.array([]), axis=-1):
        pass


class ILinearDiscreteOperator(IDiscreteOperator):

    @core.interfaces.abstractmethod
    def assemble(self, mu=np.array([])):
        assert mu.size == self.parameter_dim,\
         ValueError('Invalid parameter dimensions (was {}, expected {})'.format(mu.size, self.parameter_dim))

    def matrix(self, mu=np.array([])):
        if np.all(self._last_mu == mu):
            return self._last_mat
        else:
            self._last_mu = mu  # TODO: add some kind of log message here
            self._last_mat = self.assemble(mu)
            return self._last_mat

    def apply(self, U, mu=np.array([])):
        assert mu.size == self.parameter_dim,\
         ValueError('Invalid parameter dimensions (was {}, expected {})'.format(mu.size, self.parameter_dim))
        return matrix(mu).dot(U)  # TODO: Check what dot really does for a sparse matrix

    _last_mu = None
    _last_mat = None
