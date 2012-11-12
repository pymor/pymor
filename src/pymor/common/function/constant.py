
import numpy as np

from .interfaces import Interface

class Constant(Interface):

    def __init__(self, value, name='common.function.constant'):
        self._value = value
        self._name = name

    def evaluate(self, x):
        if type(x) is np.ndarray:
            return self._value * np.ones(x.shape)
        else:
            raise TypeError('in pymor.' + self._name + ': x has to be a numpy.array!')

    def name(self):
        return self._name
