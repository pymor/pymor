from __future__ import absolute_import, division, print_function

import numpy as np

import pymor.core as core
from pymor.tools import Named
from pymor.parameters import Parametric

class DiscreteOperatorInterface(core.BasicInterface, Parametric, Named):

    dim_source = 0
    dim_range = 0

    @core.interfaces.abstractmethod
    def apply(self, U, mu={}):
        pass

    def apply2(self, V, U, mu={}, product=None, pairwise=True):
        ''' Treat the operator as a 2-form by calculating (V, A(U)).
        If product is None, we take the standard L2-scalar product. I.e.
        if A is given by multiplication with a matrix B, then
            A.apply2(V, U) = V^T*B*U
        '''
        AU = self.apply(U, mu)
        assert not pairwise or AU.shape == V.shape
        if product is None:
            if pairwise:
                return np.sum(V * AU, axis=-1)
            else:
                return np.dot(V, AU.T)                              # use of np.dot for ndarrays?!
        elif isinstance(product, DiscreteOperatorInterface):
            return product.apply2(V, AU, pairwise)
        else:
            if pairwise:
                return np.sum(V * np.dot(product, AU.T).T, axis=-1)
            else:
                return np.dot(V, np.dot(product, AU.T))

    def __str__(self):
        return '{}: R^{} --> R^{}  (parameter type: {}, class: {})'.format(
                                           self.name, self.dim_source, self.dim_range, self.parameter_type,
                                           self.__class__.__name__)


class LinearDiscreteOperatorInterface(DiscreteOperatorInterface):

    @core.interfaces.abstractmethod
    def assemble(self, mu={}):
        pass

    def matrix(self, mu={}):
        mu = self.parse_parameter(mu)
        if self._last_mu is not None and self._last_mu.allclose(mu):
            return self._last_mat
        else:
            self._last_mu = mu.copy() # TODO: add some kind of log message here
            self._last_mat = self.assemble(mu)
            return self._last_mat

    def apply(self, U, mu={}):
        return self.matrix(mu).dot(U.T).T  # TODO: Check what dot really does for a sparse matrix

    _last_mu = None
    _last_mat = None
