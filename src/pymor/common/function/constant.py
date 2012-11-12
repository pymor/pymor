
from .interfaces import Interface

class Constant(Interface):
    def __init__(self, a):
        self._a = a
    def evaluate(self, x):
        return self._a
