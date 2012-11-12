
from .interfaces import Interface

class Constant(Interface):

    def __init__(self, value, name='common.function.constant'):
        self._value = value
        self._name = name

    def evaluate(self, x):
        return self._value

    def name(self, x):
        return self._name
